from dataclasses import dataclass, field
from typing import Callable

import torch as th

@dataclass
class ActorCriticConfig:
    device: th.device | str = "cpu"
    regularizers: list[Callable[..., th.Tensor | float]] = field(default_factory=list)
    reg_lams: list[float] = field(default_factory=list)
    num_actions: int = 2
    clip_grad: bool = True
    grad_norm_clip: float = 1.0 
    use_bias: bool = True
    value_criterion: Callable = field(default_factory=th.nn.MSELoss)
    value_lambda: float = 0.1
    value_targets: str = "td"
    gamma: float = 0.99
    advantage_bootstrap: bool = True
    off_policy_iterations: int = 0
    ppo_clipping: float = 0.1
    use_entropy: bool = False
    entropy_max_lambda: float = 0.0
    entropy_min_lambda: float = 0.0
    entropy_anneal_steps: int = 1
    normalize_advantages: bool = False


class ActorCritic:

    def __init__(
        self,
        actor: th.nn.Module,
        optimizer: th.optim.Optimizer,
        config: ActorCriticConfig,
       ):
        
        device = th.device(config.device)
        self.actor = actor
        self.optimizer = optimizer
        self.all_parameters = list(actor.parameters())
        self.actor.to(device)
        self.device = device
        self.config = config

        if len(config.regularizers) != len(config.reg_lams):
            raise ValueError("Length of regularizers and reg_lams lists must match.")
        
        if not config.use_bias and config.advantage_bootstrap:
            raise ValueError("advantage_bootstrap=True requires use_bias=True.")
        
        if config.value_targets not in {"returns", "td"}:
            raise ValueError("value_targets must be either 'returns' or 'td'.")

        if config.use_entropy and config.entropy_anneal_steps <= 1:
            raise ValueError("entropy_anneal_steps must be > 1 when use_entropy=True.")
        if config.use_entropy and config.entropy_max_lambda < config.entropy_min_lambda:
            raise ValueError("entropy_max_lambda must be >= entropy_min_lambda.")

        self.entropy_step = 0
        
    def _value_loss(
        self,
        returns: th.Tensor,
        rewards: th.Tensor,
        dones: th.Tensor,
        values: th.Tensor,
        next_values: th.Tensor,
    ) -> th.Tensor:
        if self.config.value_targets == "returns":
            targets = returns
        elif self.config.value_targets == "td":
            targets = rewards + self.config.gamma * (~dones * next_values)
        else:
            raise ValueError(f"Unknown value_targets: {self.config.value_targets}")

        return self.config.value_criterion(values, targets)
    
    def _get_policy(self, logits: th.Tensor, actions: th.Tensor) -> th.Tensor:
        probs = th.nn.functional.softmax(logits, dim=-1)
        pi = probs.gather(dim=-1, index=actions)
        return pi

    def _entropy_lambda(self) -> float:
        progress = min(self.entropy_step / (self.config.entropy_anneal_steps - 1), 1.0)

        return (
            self.config.entropy_max_lambda
            + progress * (self.config.entropy_min_lambda - self.config.entropy_max_lambda)
        )

    def _entropy_loss(self, logits: th.Tensor) -> th.Tensor:
        entropy_lambda = self._entropy_lambda()
        probs = th.nn.functional.softmax(logits, dim=-1)
        log_probs = th.nn.functional.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return -entropy_lambda * entropy
        
    def _advantages(
            self, 
            returns: th.Tensor, 
            rewards: th.Tensor,
            dones: th.Tensor,
            values: th.Tensor,
            next_values: th.Tensor
        ) -> th.Tensor:
        advantages = None
        if self.config.advantage_bootstrap:
            advantages = rewards + self.config.gamma * (~dones * next_values)
        else: 
            advantages = returns
        if self.config.use_bias:
            advantages = advantages - values.detach()

        if self.config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        return advantages

    def _policy_loss(self, pi: th.Tensor, advantage: th.Tensor) -> th.Tensor:

        if self.old_pi is None:
            self.old_pi = pi.detach()
            return -th.mean(pi.clamp_min(1e-8).log() * advantage.detach())
        else:
            ratios = pi / self.old_pi.detach()
            loss = advantage.detach() * ratios
            ppo_loss = th.clamp(ratios, 1-self.config.ppo_clipping, 1+self.config.ppo_clipping) * advantage.detach()
            loss = th.min(loss, ppo_loss)
            return -th.mean(loss)

    def train(
        self,
        rewards: th.Tensor,      # float32, [B,1]
        dones: th.Tensor,        # bool or float(0/1), [B,1]
        states: th.Tensor,       # float32, [B, obs_dim] or [B,C,H,W]
        actions: th.Tensor,      # int64, [B,1]
        next_states: th.Tensor,  # float32, same as states
        returns: th.Tensor,      # float32, [B,1]
    ) -> float:
        
        self.actor.train(True)
        need_next_values = (
            self.config.advantage_bootstrap
            or self.config.value_targets == "td"
        )

        self.old_pi, loss_sum = None, 0

        for _ in range(1 + self.config.off_policy_iterations):
            # 1. Compute model policy, values and next_values
            output: th.Tensor  = self.actor(states)
            logits = output[:, :self.config.num_actions]
            values = output[:, self.config.num_actions:self.config.num_actions + 1]

            next_values = None
            if need_next_values:
                with th.no_grad():
                    next_output: th.Tensor  = self.actor(next_states)
                    next_values = next_output[:, self.config.num_actions:self.config.num_actions + 1]
            pi = self._get_policy(logits, actions)

            # 2. Compute losses
            policy_loss = self._policy_loss(
                pi, 
                self._advantages(
                    returns, 
                    rewards,
                    dones,
                    values,
                    next_values
                )
            )
            entropy_loss = 0.0
            if self.config.use_entropy:
                entropy_loss = self._entropy_loss(logits)
            value_loss = 0.0
            if self.config.use_bias:
                value_loss = self.config.value_lambda * self._value_loss(
                    returns,
                    rewards,
                    dones,
                    values,
                    next_values
                )
            reg_loss = (
                sum(
                    lam * reg(self.actor, rewards, dones, states, actions, next_states)
                    for reg, lam in zip(self.config.regularizers, self.config.reg_lams)
                )
                if self.config.regularizers else 0.0
            )
            loss = policy_loss + value_loss + entropy_loss + reg_loss

            # 3. Optimize
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.clip_grad:
                th.nn.utils.clip_grad_norm_(self.all_parameters, self.config.grad_norm_clip)
            self.optimizer.step()
            loss_sum += loss.item()

        self.entropy_step += 1
        return loss_sum
