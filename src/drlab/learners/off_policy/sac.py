import torch as th
import torch.nn.functional as F
from torch.distributions import Normal

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

        self.config = config
        self.device = th.device(config.device)
        self.gamma = config.gamma
        self.regularizers = config.regularizers
        self.reg_lams = config.reg_lams
        self.clip_grad = config.clip_grad
        self.grad_norm_clip = config.grad_norm_clip
        self.target_update_calls = 0
        self.use_target_model = config.use_target_model
        self.target_update = config.target_update
        self.target_update_interval = config.target_update_interval
        self.soft_target_update_param = config.soft_target_update_param
        self._validate_config()

        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)
        self.critic1_target = self._make_target_model(self.critic1)
        self.critic2_target = self._make_target_model(self.critic2)

        self.actor_optimizer = actor_optimizer
        self.critic1_optimizer = critic1_optimizer
        self.critic2_optimizer = critic2_optimizer
        self.model = self.actor
        self.optimizer = self.actor_optimizer
        self.parameters = list(self.actor.parameters())

        self.target_entropy = self.config.target_entropy
        if self.target_entropy is None:
            self.target_entropy = -float(self.config.action_dim)

        self.log_alpha = th.nn.Parameter(th.tensor(0.0, device=self.device))
        self.alpha_optimizer = th.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.all_parameters = (
            self.parameters
            + list(self.critic1.parameters())
            + list(self.critic2.parameters())
            + [self.log_alpha]
        )
        self.last_losses: dict[str, float] = {}

    @property
    def alpha(self) -> th.Tensor:
        return self.log_alpha.exp()

    def get_policy_dist(self, states: th.Tensor):
        output = self.actor(states)
        mean = output[:, : self.config.action_dim]
        log_std = output[:, self.config.action_dim :]
        log_std = th.clamp(log_std, self.config.min_log_std, self.config.max_log_std)
        std = th.exp(log_std)
        return mean, std

    def sample_action_and_log_prob(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor]:
        mean, std = self.get_policy_dist(obs)

        if deterministic:
            u = mean
            return th.tanh(u), th.zeros(obs.shape[0], 1, device=obs.device)
        else:
            dist = Normal(mean, std)
            u = dist.rsample()  # reparameterization trick
            action = th.tanh(u)

            # Gaussian log prob before tanh
            log_prob = dist.log_prob(u)

            # Tanh correction
            log_prob -= th.log(1.0 - action.pow(2) + 1e-6)

            # Sum over action dimensions
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def train(
        self,
        rewards: th.Tensor,      # float32, [B,1]
        dones: th.Tensor,        # bool or float(0/1), [B,1]
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

            target_q1 = self.critic1_target(th.cat([next_states, next_actions], dim=-1))
            target_q2 = self.critic2_target(th.cat([next_states, next_actions], dim=-1))
            target_q = th.min(target_q1, target_q2)

            not_done = 1.0 - dones.float()
            targets = rewards + self.config.gamma * (
                not_done * (target_q - self.alpha.detach() * next_log_probs)
            )

        current_q1 = self.critic1(th.cat([states, actions], dim=-1))
        current_q2 = self.critic2(th.cat([states, actions], dim=-1))

        critic1_loss = F.mse_loss(current_q1, targets)
        critic2_loss = F.mse_loss(current_q2, targets)
        critic_loss = critic1_loss + critic2_loss
        self.optimize(
            critic1_loss,
            optimizer=self.critic1_optimizer,
            parameters=self.critic1.parameters(),
        )
        self.optimize(
            critic2_loss,
            optimizer=self.critic2_optimizer,
            parameters=self.critic2.parameters(),
        )

        # --------------------------------------------------
        # 2. Update actor
        # --------------------------------------------------
        new_actions, log_probs = self.sample_action_and_log_prob(states)
        new_q1 = self.critic1(th.cat([states, new_actions], dim=-1))
        new_q2 = self.critic2(th.cat([states, new_actions], dim=-1))
        new_q = th.min(new_q1, new_q2)

        actor_loss = (self.alpha.detach() * log_probs - new_q).mean()
        self.optimize(
            actor_loss,
            optimizer=self.actor_optimizer,
            parameters=self.actor.parameters(),
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
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)

        loss = actor_loss + critic_loss + alpha_loss
        self.last_losses = {
            "actor": float(actor_loss.item()),
            "critic": float(critic_loss.item()),
            "alpha": float(alpha_loss.item()),
            "total": float(loss.item()),
        }

        return self.last_losses["total"]
