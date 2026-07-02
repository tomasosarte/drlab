import torch as th

from .base import OffPolicyLearner
from .configs import DQNConfig


class DQNLearner(OffPolicyLearner):
    config: DQNConfig

    def __init__(
        self,
        model: th.nn.Module,
        optimizer: th.optim.Optimizer,
        config: DQNConfig,
    ):
        super().__init__(model, optimizer, config)

        self.criterion = config.criterion
        self.double_q = config.double_q
        self.num_actions = config.num_actions

    def _validate_config(self):
        super()._validate_config()

        if self.config.double_q and not self.config.use_target_model:
            raise ValueError(
                "double_q=True requires use_target_model=True "
                "(needs a target network)."
            )

        if self.config.num_actions <= 0:
            raise ValueError(f"num_actions must be > 0, got {self.config.num_actions}")

    def q_values(self, states: th.Tensor, target: bool = False) -> th.Tensor:
        return self.forward(states, target=target)[:, : self.num_actions]

    def _current_values(self, states: th.Tensor, actions: th.Tensor):
        qvalues = self.q_values(states, target=False)
        return qvalues.gather(dim=-1, index=actions)

    def _next_values(self, next_states: th.Tensor) -> th.Tensor:
        with th.no_grad():
            if self.double_q:
                # action selection from online net, evaluation from target net
                online_q = self.q_values(next_states, target=False)
                actions = online_q.argmax(dim=-1, keepdim=True)
                target_q = self.q_values(next_states, target=True)
                return target_q.gather(dim=-1, index=actions)
            else:
                q = self.q_values(next_states, target=self.use_target_model)
                return q.max(dim=-1, keepdim=True)[0]

    def train(
        self,
        rewards: th.Tensor,      # float32, [B,1]
        dones: th.Tensor,        # bool or float(0/1), [B,1]
        states: th.Tensor,       # float32, [B, obs_dim] or [B,C,H,W]
        actions: th.Tensor,      # int64, [B,1]
        next_states: th.Tensor,  # float32, same as states
    ) -> float:
        self.model.train(True)

        # Compute targets and loss
        not_done = 1.0 - dones.float()
        targets = rewards + self.gamma * (not_done * self._next_values(next_states))
        pred = self._current_values(states, actions)
        td_loss = self.criterion(pred, targets)
        reg_loss = self._regularization_loss(
            rewards=rewards,
            dones=dones,
            states=states,
            actions=actions,
            next_states=next_states,
        )
        loss = td_loss + reg_loss

        # Optimize
        self.optimize(loss)

        self.target_model_update()
        return float(loss.item())
