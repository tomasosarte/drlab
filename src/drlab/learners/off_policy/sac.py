import math

import torch as th

from .base import OffPolicyLearner
from .configs import SACConfig


class SACLearner(OffPolicyLearner):
    """Soft Actor-Critic for continuous action spaces.

    The actor outputs `[mean, log_std]` with `2 * action_dim` columns. Critics
    receive `cat([state, action], dim=-1)` and output one scalar Q-value.
    """

    config: SACConfig

    def __init__(
        self,
        actor: th.nn.Module,
        q1: th.nn.Module,
        q2: th.nn.Module,
        actor_optimizer: th.optim.Optimizer,
        q1_optimizer: th.optim.Optimizer,
        q2_optimizer: th.optim.Optimizer,
        config: SACConfig,
    ):
        if q1 is q2:
            raise ValueError("q1 and q2 must be separate critic modules.")

        super().__init__(q1, q1_optimizer, config)

        self.actor = actor.to(self.device)
        self.q1 = self.model
        self.q2 = q2.to(self.device)
        self.target_q1 = self.target_model
        self.target_q2 = self._make_target_model(self.q2)

        self.actor_optimizer = actor_optimizer
        self.q1_optimizer = self.optimizer
        self.q2_optimizer = q2_optimizer

        self.action_dim = config.action_dim
        self.criterion = config.criterion
        self.target_entropy = float(config.target_entropy)
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

        self.action_low = self._action_bound(config.action_low, "action_low")
        self.action_high = self._action_bound(config.action_high, "action_high")
        if not th.all(self.action_high > self.action_low):
            raise ValueError("action_high must be greater than action_low.")
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0

        self.actor_parameters = list(self.actor.parameters())
        self.q1_parameters = list(self.q1.parameters())
        self.q2_parameters = list(self.q2.parameters())
        self.critic_parameters = self.q1_parameters + self.q2_parameters
        self.parameters = (
            self.actor_parameters + self.q1_parameters + self.q2_parameters
        )
        self.all_parameters = self.parameters

        self.log_alpha = th.nn.Parameter(
            th.tensor(
                math.log(config.alpha),
                dtype=th.float32,
                device=self.device,
            ),
            requires_grad=config.auto_entropy_tuning,
        )
        self.alpha_optimizer = None
        if config.auto_entropy_tuning:
            self.alpha_optimizer = th.optim.Adam([self.log_alpha], lr=config.alpha_lr)

        self.last_metrics: dict[str, float] = {}

    @property
    def alpha(self) -> th.Tensor:
        return self.log_alpha.exp()

    def _validate_config(self):
        super()._validate_config()

        if not self.config.use_target_model:
            raise ValueError("SAC requires use_target_model=True for target critics.")

        if self.config.action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {self.config.action_dim}")

        if self.config.log_std_min >= self.config.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max.")

        if self.config.alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {self.config.alpha}")

        if self.config.target_entropy is None or not math.isfinite(
            self.config.target_entropy
        ):
            raise ValueError("target_entropy must be finite.")

        if self.config.auto_entropy_tuning and self.config.alpha_lr <= 0.0:
            raise ValueError(f"alpha_lr must be > 0, got {self.config.alpha_lr}")

        if self.config.min_log_alpha > self.config.max_log_alpha:
            raise ValueError("min_log_alpha must be <= max_log_alpha.")

    def _target_model_pairs(self):
        return [
            (self.target_q1, self.q1),
            (self.target_q2, self.q2),
        ]

    def _action_bound(self, value, name: str) -> th.Tensor:
        bound = th.as_tensor(value, dtype=th.float32, device=self.device)
        if bound.ndim == 0:
            bound = bound.repeat(self.action_dim)
        if tuple(bound.shape) != (self.action_dim,):
            raise ValueError(
                f"{name} must be scalar or shape ({self.action_dim},), "
                f"got {tuple(bound.shape)}"
            )
        return bound.view(1, self.action_dim)

    def _actor_stats(self, states: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        output = self.actor(states)
        expected = 2 * self.action_dim
        if output.ndim < 2 or output.shape[-1] < expected:
            raise ValueError(
                f"actor must output at least {expected} columns, "
                f"got shape {tuple(output.shape)}"
            )
        mean = output[:, : self.action_dim]
        log_std = output[:, self.action_dim : expected]
        log_std = th.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample_actions(
        self,
        states: th.Tensor,
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor]:
        mean, log_std = self._actor_stats(states)
        if deterministic:
            raw_action = mean
        else:
            raw_action = th.distributions.Normal(mean, log_std.exp()).rsample()

        squashed_action = th.tanh(raw_action)
        action = squashed_action * self.action_scale + self.action_bias

        log_prob = None
        if not deterministic:
            normal = th.distributions.Normal(mean, log_std.exp())
            log_prob = normal.log_prob(raw_action)
            log_prob -= th.log(self.action_scale * (1 - squashed_action.pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def act(self, states: th.Tensor, deterministic: bool = False) -> th.Tensor:
        with th.no_grad():
            actions, _ = self.sample_actions(states, deterministic=deterministic)
        return actions

    def _critic_input(self, states: th.Tensor, actions: th.Tensor) -> th.Tensor:
        if states.ndim > 2:
            states = states.flatten(start_dim=1)
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        return th.cat([states, actions], dim=-1)

    def _critic_value(
        self,
        critic: th.nn.Module,
        states: th.Tensor,
        actions: th.Tensor,
    ) -> th.Tensor:
        values = critic(self._critic_input(states, actions))
        if values.ndim == 1:
            values = values.unsqueeze(-1)
        return values[:, :1]

    def q_values(
        self,
        states: th.Tensor,
        actions: th.Tensor,
        target: bool = False,
    ) -> tuple[th.Tensor, th.Tensor]:
        q1 = self.target_q1 if target else self.q1
        q2 = self.target_q2 if target else self.q2
        return (
            self._critic_value(q1, states, actions),
            self._critic_value(q2, states, actions),
        )

    def forward(
        self,
        states: th.Tensor,
        actions: th.Tensor,
        target: bool = False,
    ) -> tuple[th.Tensor, th.Tensor]:
        return self.q_values(states, actions, target=target)

    def _critic_loss(
        self,
        rewards: th.Tensor,
        dones: th.Tensor,
        states: th.Tensor,
        actions: th.Tensor,
        next_states: th.Tensor,
    ) -> th.Tensor:
        with th.no_grad():
            next_actions, next_log_probs = self.sample_actions(next_states)
            next_q1, next_q2 = self.q_values(next_states, next_actions, target=True)
            next_q = th.minimum(next_q1, next_q2) - self.alpha.detach() * next_log_probs
            targets = rewards + self.gamma * (1.0 - dones.float()) * next_q

        q1, q2 = self.q_values(states, actions)
        q1_loss = self.criterion(q1, targets)
        q2_loss = self.criterion(q2, targets)
        reg_loss = self._regularization_loss(
            rewards,
            dones,
            states,
            actions,
            next_states,
            model=self.q1,
        ) + self._regularization_loss(
            rewards,
            dones,
            states,
            actions,
            next_states,
            model=self.q2,
        )
        return q1_loss + q2_loss + reg_loss

    def _actor_loss(self, states: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        actions, log_probs = self.sample_actions(states)
        q1, q2 = self.q_values(states, actions)
        q = th.minimum(q1, q2)
        return (self.alpha.detach() * log_probs - q).mean(), log_probs

    def _alpha_loss(self, log_probs: th.Tensor) -> th.Tensor:
        if not self.config.auto_entropy_tuning:
            return log_probs.new_tensor(0.0)
        return -(
            self.log_alpha * (log_probs.detach() + self.target_entropy)
        ).mean()

    def _optimize_alpha(self, alpha_loss: th.Tensor):
        if self.alpha_optimizer is None:
            return

        self.optimize(alpha_loss, self.alpha_optimizer, [self.log_alpha])
        with th.no_grad():
            self.log_alpha.clamp_(
                min=self.config.min_log_alpha,
                max=self.config.max_log_alpha,
            )

    def train(
        self,
        rewards: th.Tensor,
        dones: th.Tensor,
        states: th.Tensor,
        actions: th.Tensor,
        next_states: th.Tensor,
    ) -> float:
        self.actor.train(True)
        self.q1.train(True)
        self.q2.train(True)

        actions = actions.float()
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)

        critic_loss = self._critic_loss(
            rewards=rewards,
            dones=dones,
            states=states,
            actions=actions,
            next_states=next_states,
        )
        self.optimize(
            critic_loss,
            [self.q1_optimizer, self.q2_optimizer],
            self.critic_parameters,
        )

        self._set_requires_grad(self.critic_parameters, False)
        actor_loss, log_probs = self._actor_loss(states)
        self.optimize(actor_loss, self.actor_optimizer, self.actor_parameters)
        self._set_requires_grad(self.critic_parameters, True)

        alpha_loss = self._alpha_loss(log_probs)
        self._optimize_alpha(alpha_loss)

        self.target_model_update()

        total_loss = critic_loss.detach() + actor_loss.detach() + alpha_loss.detach()
        entropy = -log_probs.detach().mean()
        self.last_metrics = {
            "loss": float(total_loss.item()),
            "critic_loss": float(critic_loss.detach().item()),
            "actor_loss": float(actor_loss.detach().item()),
            "alpha_loss": float(alpha_loss.detach().item()),
            "alpha": float(self.alpha.detach().item()),
            "entropy": float(entropy.item()),
        }
        return self.last_metrics["loss"]
