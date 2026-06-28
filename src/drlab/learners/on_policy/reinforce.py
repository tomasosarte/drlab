import torch as th

from .base import OnPolicyLearner
from .configs import ReinforceConfig

class ReinforceLearner(OnPolicyLearner):
    config: ReinforceConfig

    def _policy_loss(
        self,
        pi: th.Tensor,
        returns: th.Tensor,
    ) -> th.Tensor:
        log_pi = pi.clamp_min(1e-8).log()
        return -(log_pi * returns.detach()).mean()

    def _advantages(
        self,
        returns: th.Tensor,
        rewards: th.Tensor,
        dones: th.Tensor,
        values: th.Tensor | None,
        next_values: th.Tensor | None,
    ) -> th.Tensor:
        advantages = returns

        if self.config.normalize_returns and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        return advantages

    def train(
        self,
        rewards: th.Tensor,
        dones: th.Tensor,
        states: th.Tensor,
        actions: th.Tensor,
        next_states: th.Tensor,
        returns: th.Tensor,
    ) -> float:
        
        self.actor.train(True)

        # 1. Get policy + advantages
        logits, _ = self.forward(states, get_values=False)
        pi = self._get_policy(logits, actions)

        advantages = self._advantages(
            returns=returns,
            rewards=rewards,
            dones=dones,
            values=None,
            next_values=None,
        )
        
        # 2. Get Loss
        policy_loss = self._policy_loss(pi, advantages)
        entropy_loss = self._entropy_loss(logits)
        reg_loss = self._regularization_loss(
            rewards=rewards,
            dones=dones,
            states=states,
            actions=actions,
            next_states=next_states,
        )
        loss = policy_loss + entropy_loss + reg_loss


        # 3. Optimize + update state
        self.optimize(loss)
        self.entropy_step += 1

        return loss.item()
