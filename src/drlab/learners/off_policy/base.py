from copy import deepcopy

import torch as th

from .configs import OffPolicyConfig, TargetUpdate


class OffPolicyLearner:
    def __init__(
        self,
        model: th.nn.Module,
        optimizer: th.optim.Optimizer,
        config: OffPolicyConfig,
    ):
        self.config = config
        self.device = th.device(config.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.parameters = list(self.model.parameters())
        self.all_parameters = self.parameters

        self.gamma = config.gamma
        self.regularizers = config.regularizers
        self.reg_lams = config.reg_lams
        self.clip_grad = config.clip_grad
        self.grad_norm_clip = config.grad_norm_clip

        self.target_update_calls = 0
        self.use_target_model = config.use_target_model
        self.target_update = TargetUpdate(config.target_update)
        self.target_update_interval = config.target_update_interval
        self.soft_target_update_param = config.soft_target_update_param

        self._validate_config()

        self.target_model = None
        if self.use_target_model:
            self.target_model = deepcopy(self.model)
            self.target_model.to(self.device)
            self.target_model.eval()
            for parameter in self.target_model.parameters():
                parameter.requires_grad = False

    def _validate_config(self):
        if len(self.regularizers) != len(self.reg_lams):
            raise ValueError("regularizers and reg_lams must have same length.")

        if self.target_update not in {TargetUpdate.HARD, TargetUpdate.SOFT}:
            raise ValueError(
                f"target_update must be 'hard' or 'soft', got {self.target_update!r}"
            )

        if self.target_update == TargetUpdate.HARD and self.target_update_interval <= 0:
            raise ValueError(
                "target_update_interval must be > 0 for hard updates, "
                f"got {self.target_update_interval}"
            )

        if self.target_update == TargetUpdate.SOFT and not (
            0.0 < self.soft_target_update_param <= 1.0
        ):
            raise ValueError("soft_target_update_param must be in (0, 1].")

    def forward(self, states: th.Tensor, target: bool = False) -> th.Tensor:
        model = (
            self.target_model
            if target and self.target_model is not None
            else self.model
        )
        return model(states)

    def target_model_update(self):
        if self.target_model is None:
            return

        if self.target_update == TargetUpdate.HARD:
            self.target_update_calls = (
                self.target_update_calls + 1
            ) % self.target_update_interval
            if self.target_update_calls != 0:
                return

            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()
            return

        if self.target_update == TargetUpdate.SOFT:
            tau = self.soft_target_update_param
            with th.no_grad():
                for target_parameter, model_parameter in zip(
                    self.target_model.parameters(),
                    self.model.parameters(),
                ):
                    target_parameter.copy_(
                        target_parameter * (1.0 - tau) + model_parameter * tau
                    )
            return

        raise KeyError(f"Target model update unknown: {self.target_update}")

    def _regularization_loss(
        self,
        rewards: th.Tensor,
        dones: th.Tensor,
        states: th.Tensor,
        actions: th.Tensor,
        next_states: th.Tensor,
    ):
        if not self.regularizers:
            return 0.0

        return sum(
            lam * regularizer(self.model, rewards, dones, states, actions, next_states)
            for regularizer, lam in zip(self.regularizers, self.reg_lams)
        )

    def optimize(self, loss: th.Tensor):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.clip_grad:
            th.nn.utils.clip_grad_norm_(
                self.parameters,
                self.grad_norm_clip,
            )
        self.optimizer.step()

    def requires_returns(self) -> bool:
        return False

    def train(
        self,
        rewards: th.Tensor,
        dones: th.Tensor,
        states: th.Tensor,
        actions: th.Tensor,
        next_states: th.Tensor,
    ) -> float:
        raise NotImplementedError
