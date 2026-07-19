import math

import torch as th

from .base import OffPolicyLearner
from .configs import SACConfig


class SACLearner(OffPolicyLearner):
    config: SACConfig

    def __init__(
        self,
        actor: th.nn.Module,
        critic1: th.nn.Module,
        critic2: th.nn.Module,
        actor_optimizer: th.optim.Optimizer,
        critic1_optimizer: th.optim.Optimizer,
        critic2_optimizer: th.optim.Optimizer,
        config: SACConfig,
    ):
        super().__init__(config)

        if not self.use_target_model:
            raise ValueError("SAC requires target critics, so use_target_model=True.")

        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.actor_params = tuple(self.actor.parameters())

        self.critic1_target = self.make_target_model(self.critic1)
        self.critic2_target = self.make_target_model(self.critic2)

        self.actor_optimizer = actor_optimizer
        self.critic1_optimizer = critic1_optimizer
        self.critic2_optimizer = critic2_optimizer

        self.criterion = config.criterion

        # Entropy tuning parameters
        self.target_entropy = config.target_entropy
        if self.target_entropy is None:
            self.target_entropy = -float(config.action_dim)

        self.log_alpha = th.nn.Parameter(
            th.tensor(math.log(config.initial_alpha), device=self.device)
        )

        self.alpha_optimizer = th.optim.Adam(
            [self.log_alpha],
            lr=config.alpha_lr,
        )

    @property
    def alpha(self) -> th.Tensor:
        return self.log_alpha.exp()

    def _get_policy_params(self, states: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        output = self.actor(states)
        mean = output[:, : self.config.action_dim]
        log_std = output[:, self.config.action_dim :]
        log_std = th.clamp(log_std, self.config.min_log_std, self.config.max_log_std)
        return mean, log_std

    def get_policy_dist(self, states: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        mean, log_std = self._get_policy_params(states)
        std = th.exp(log_std)
        return mean, std

    def sample_action_and_log_prob(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor]:
        mean, log_std = self._get_policy_params(obs)

        if deterministic:
            u = mean
            return th.tanh(u), th.zeros(obs.shape[0], 1, device=obs.device)

        # Specialized reparameterized Gaussian sampling avoids the generic
        # distribution object's redundant scale and normalized-noise work.
        eps = th.randn_like(mean)
        u = mean + th.exp(log_std) * eps
        action = th.tanh(u)

        log_prob = -0.5 * (
            eps.square() + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_prob -= th.log(1.0 - action.pow(2) + 1e-6)  # Tanh correction
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # Sum over action dimensions

        return action, log_prob

    def train(
        self,
        rewards: th.Tensor,      # float32, [B,1]
        terminated: th.Tensor,   # bool or float(0/1), [B,1]
        states: th.Tensor,       # float32, [B, obs_dim] or [B,C,H,W]
        actions: th.Tensor,      # float32, [B, action_dim]
        next_states: th.Tensor,  # float32, same as states
    ) -> float:
        self.actor.train(True)
        self.critic1.train(True)
        self.critic2.train(True)

        # --------------------------------------------------
        # 1. Update critics
        # --------------------------------------------------
        with th.no_grad():
            next_actions, next_log_probs = self.sample_action_and_log_prob(next_states)

            next_action_states = th.cat([next_states, next_actions], dim=-1)
            target_q1 = self.critic1_target(next_action_states)
            target_q2 = self.critic2_target(next_action_states)
            target_q = th.min(target_q1, target_q2)

            not_terminated = 1.0 - terminated.float()
            targets = rewards + self.config.gamma * (
                not_terminated * (target_q - self.alpha.detach() * next_log_probs)
            )

        action_states = th.cat([states, actions], dim=-1)
        current_q1 = self.critic1(action_states)
        current_q2 = self.critic2(action_states)

        critic1_loss = self.criterion(current_q1, targets)
        critic2_loss = self.criterion(current_q2, targets)
        critic_loss = critic1_loss + critic2_loss
        self.critic1_optimizer.zero_grad(set_to_none=True)
        self.critic2_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()

        if self.clip_grad:
            # Clip each critic independently to preserve the previous behavior.
            th.nn.utils.clip_grad_norm_(
                self.critic1.parameters(),
                self.grad_norm_clip,
            )
            th.nn.utils.clip_grad_norm_(
                self.critic2.parameters(),
                self.grad_norm_clip,
            )

        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # --------------------------------------------------
        # 2. Update actor
        # --------------------------------------------------
        new_actions, log_probs = self.sample_action_and_log_prob(states)
        new_action_states = th.cat([states, new_actions], dim=-1)
        new_q1 = self.critic1(new_action_states)
        new_q2 = self.critic2(new_action_states)
        new_q = th.min(new_q1, new_q2)

        policy_loss = (self.alpha.detach() * log_probs - new_q).mean()
        reg_loss = self.regularization_loss(
            self.actor,
            rewards=rewards,
            terminated=terminated,
            states=states,
            actions=actions,
            next_states=next_states,
        )
        actor_loss = policy_loss + reg_loss
        self.optimize(
            actor_loss,
            optimizer=self.actor_optimizer,
            parameters=self.actor_params,
        )

        # --------------------------------------------------
        # 3. Update alpha
        # --------------------------------------------------
        alpha_loss = -(
            self.log_alpha * (log_probs + self.target_entropy).detach()
        ).mean()

        self.optimize(
            alpha_loss,
            optimizer=self.alpha_optimizer,
            parameters=[self.log_alpha],
        )

        # Update target networks
        self.update_targets([
            (self.critic1_target, self.critic1),
            (self.critic2_target, self.critic2),
        ])

        loss = actor_loss + critic_loss + alpha_loss
        self.last_losses = {
            "actor": float(policy_loss.item()),
            "critic": float(critic_loss.item()),
            "alpha": float(alpha_loss.item()),
            "regularization": float(reg_loss.item()),
            "total": float(loss.item()),
        }

        return self.last_losses["total"]
