import torch as th

from .actor_critic import ActorCriticLearner
from .configs import PPOConfig, ValueTargets


class PPOLearner(ActorCriticLearner):
    config: PPOConfig

    def _validate_config(self):
        super()._validate_config()

        if self.config.ppo_clipping <= 0:
            raise ValueError("ppo_clipping must be > 0.")

        if self.config.ppo_iterations < 1:
            raise ValueError("ppo_iterations must be >= 1.")
        
    def _policy_loss(
        self,
        pi: th.Tensor,
        old_pi: th.Tensor,
        advantages: th.Tensor,
    ) -> th.Tensor:
        ratios = pi / old_pi.clamp_min(1e-8)

        unclipped = ratios * advantages.detach()

        clipped = th.clamp(
            ratios,
            1.0 - self.config.ppo_clipping,
            1.0 + self.config.ppo_clipping,
        ) * advantages.detach()

        return -th.min(unclipped, clipped).mean()
    

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

        # 1. Store old policy before updates
        with th.no_grad():
            old_logits, old_values = self.forward(
                states,
                get_values=self.config.use_bias,
            )

            old_pi = self._get_policy(old_logits, actions)

            need_next_values = (
                self.config.use_bias
                and (
                    self.config.advantage_bootstrap
                    or self.config.value_targets == ValueTargets.TD
                )
            )
            next_values = None
            if need_next_values:
                _, next_values = self.forward(next_states, get_values=True)

            advantages = self._advantages(
                returns=returns,
                rewards=rewards,
                dones=dones,
                values=old_values,
                next_values=next_values,
            )

        loss_sum = 0.0

        # 2. Reuse same rollout batch for several PPO epochs
        for _ in range(self.config.ppo_iterations):
            logits, values = self.forward(
                states,
                get_values=self.config.use_bias,
            )

            pi = self._get_policy(logits, actions)

            policy_loss = self._policy_loss(
                pi=pi,
                old_pi=old_pi,
                advantages=advantages,
            )

            value_loss = logits.new_tensor(0.0)
            if self.config.use_bias:
                value_loss = self.config.value_lambda * self._value_loss(
                    returns=returns,
                    rewards=rewards,
                    dones=dones,
                    values=values,
                    next_values=next_values,
                )

            entropy_loss = self._entropy_loss(logits)

            reg_loss = self._regularization_loss(
                rewards=rewards,
                dones=dones,
                states=states,
                actions=actions,
                next_states=next_states,
            )

            loss = policy_loss + value_loss + entropy_loss + reg_loss

            self.optimize(loss)

            loss_sum += loss.item()

        self.entropy_step += 1

        return loss_sum
