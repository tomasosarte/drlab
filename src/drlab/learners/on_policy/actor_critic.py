import torch as th

from .base import OnPolicyLearner
from .configs import ActorCriticConfig, ValueTargets

class ActorCriticLearner(OnPolicyLearner):
    config: ActorCriticConfig

    def _validate_config(self):
        super()._validate_config()

        if not isinstance(self.config.value_targets, ValueTargets):
            raise ValueError("value_targets must be a ValueTargets enum.")

        if not self.config.use_bias and self.config.advantage_bootstrap:
            raise ValueError("advantage_bootstrap=True requires use_bias=True.")

    def requires_returns(self) -> bool:
        return (
            not self.config.advantage_bootstrap
            or (
                self.config.use_bias
                and self.config.value_targets == ValueTargets.RETURNS
            )
        )

    def _value_loss(
        self,
        returns: th.Tensor,
        rewards: th.Tensor,
        terminated: th.Tensor,
        values: th.Tensor,
        next_values: th.Tensor,
    ) -> th.Tensor:
        if self.config.value_targets == ValueTargets.RETURNS:
            targets = returns
        elif self.config.value_targets == ValueTargets.TD:
            targets = rewards + self.config.gamma * (~terminated * next_values)
        else:
            raise ValueError(f"Unknown value_targets: {self.config.value_targets}")
        return self.config.value_criterion(values, targets)
    
    def _advantages(
            self, 
            returns: th.Tensor, 
            rewards: th.Tensor,
            terminated: th.Tensor,
            values: th.Tensor,
            next_values: th.Tensor
        ) -> th.Tensor:
        advantages = None
        if self.config.advantage_bootstrap:
            advantages = rewards + self.config.gamma * (~terminated * next_values)
        else: 
            advantages = returns
        if self.config.use_bias:
            advantages = advantages - values.detach()

        if self.config.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        return advantages
    
    def _policy_loss(
        self,
        pi: th.Tensor,
        advantages: th.Tensor,
    ) -> th.Tensor:
        log_pi = pi.clamp_min(1e-8).log()
        return -(log_pi * advantages.detach()).mean()

    def train(
        self,
        rewards: th.Tensor,
        terminated: th.Tensor,
        states: th.Tensor,
        actions: th.Tensor,
        next_states: th.Tensor,
        returns: th.Tensor,
    ) -> float:
        
        self.actor.train(True)

        # 1. Forward current states
        logits, values = self.forward(states, get_values=self.config.use_bias)

        # 2. Forward next states if needed
        need_next_values = (
            self.config.use_bias
            and (
                self.config.advantage_bootstrap
                or self.config.value_targets == ValueTargets.TD
            )
        )
        next_values = None
        if need_next_values:
            with th.no_grad():
                _, next_values = self.forward(next_states, get_values=True)

        # 3. Policy
        pi = self._get_policy(logits, actions)

        # 4. Advantage
        advantages = self._advantages(
            returns=returns,
            rewards=rewards,
            terminated=terminated,
            values=values,
            next_values=next_values,
        )

        # 5. Losses
        policy_loss = self._policy_loss(pi, advantages)

        value_loss = logits.new_tensor(0.0)
        if self.config.use_bias:
            value_loss = self.config.value_lambda * self._value_loss(
                returns=returns,
                rewards=rewards,
                terminated=terminated,
                values=values,
                next_values=next_values,
            )

        entropy_loss = self._entropy_loss(logits)

        reg_loss = self._regularization_loss(
            rewards=rewards,
            terminated=terminated,
            states=states,
            actions=actions,
            next_states=next_states,
        )

        loss = policy_loss + value_loss + entropy_loss + reg_loss

        # 6. Optimize
        self.optimize(loss)
        self.entropy_step += 1

        return loss.item()
