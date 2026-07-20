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
        super().__init__(config)

        self.model = model.to(self.device)
        self.optimizer = optimizer

        self.criterion = config.criterion
        self.double_q = config.double_q
        self.num_actions = config.num_actions

        self._validate_dqn_config()

        self.target_model = None
        if self.use_target_model:
            self.target_model = self.make_target_model(self.model)

    def _validate_dqn_config(self) -> None:
        if self.double_q and not self.use_target_model:
            raise ValueError("double_q=True requires use_target_model=True.")

        if self.num_actions <= 0:
            raise ValueError(f"num_actions must be > 0, got {self.num_actions}.")

    def q_values(self, states: th.Tensor, target: bool = False) -> th.Tensor:
        model = self.target_model if target and self.target_model is not None else self.model
        return model(states)[:, : self.num_actions]

    def current_values(self, states: th.Tensor, actions: th.Tensor):
        qvalues = self.q_values(states, target=False)
        return qvalues.gather(dim=-1, index=actions)

    def next_values(self, next_states: th.Tensor) -> th.Tensor:
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
        terminated: th.Tensor,   # bool or float(0/1), [B,1]
        states: th.Tensor,       # float32, [B, obs_dim] or [B,C,H,W]
        actions: th.Tensor,      # int64, [B,1]
        next_states: th.Tensor,  # float32, same as states
    ) -> float:
        self.model.train(True)

        # Compute targets
        not_terminated = 1.0 - terminated.float()
        targets = rewards + self.gamma * (
            not_terminated * self.next_values(next_states)
        )
        pred = self.current_values(states, actions)
        
        # Compute loss
        td_loss = self.criterion(pred, targets)
        reg_loss = self.regularization_loss(
            self.model,
            rewards=rewards,
            terminated=terminated,
            states=states,
            actions=actions,
            next_states=next_states,
        )
        loss = td_loss + reg_loss

        # Update learner state
        self.optimize(
            loss,
            optimizer=self.optimizer,
            parameters=self.model.parameters(),
            grad_norm_clip=self.grad_norm_clip,
        )

        if self.target_model is not None:
            self.update_targets([(self.target_model, self.model)])

        self.last_losses = {
            "td": float(td_loss.item()),
            "regularization": float(reg_loss.item()),
            "total": float(loss.item()),
        }

        return self.last_losses["total"]
