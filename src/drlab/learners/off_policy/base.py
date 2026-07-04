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
            self.target_model = self._make_target_model(self.model)

    def _validate_config(self):
        if len(self.regularizers) != len(self.reg_lams):
            raise ValueError("regularizers and reg_lams must have same length.")

        if self.target_update not in {TargetUpdate.HARD, TargetUpdate.SOFT}:
            raise ValueError(
                f"target_update must be 'hard' or 'soft', got {self.target_update!r}"
            )

        if self.target_update == TargetUpdate.HARD and self.target_update_interval <= 0:
            raise ValueError("target_update_interval must be > 0 for hard updates.")

        if self.target_update == TargetUpdate.SOFT and not (
            0.0 < self.soft_target_update_param <= 1.0
        ):
            raise ValueError("soft_target_update_param must be in (0, 1].")

    def _make_target_model(self, model: th.nn.Module) -> th.nn.Module:
        target = deepcopy(model).to(self.device)
        target.eval()

        for parameter in target.parameters():
            parameter.requires_grad = False

        return target

    def _hard_update(self, target_model: th.nn.Module, model: th.nn.Module):
        target_model.load_state_dict(model.state_dict())
        target_model.eval()

    def _soft_update(
        self,
        target_model: th.nn.Module,
        model: th.nn.Module,
    ):
        tau = self.soft_target_update_param

        with th.no_grad():
            for target_parameter, model_parameter in zip(
                target_model.parameters(),
                model.parameters(),
            ):
                target_parameter.copy_(
                    (1.0 - tau) * target_parameter + tau * model_parameter
                )

    def _hard_update_due(self) -> bool:
        self.target_update_calls = (
            self.target_update_calls + 1
        ) % self.target_update_interval

        return self.target_update_calls == 0

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
            if self._hard_update_due():
                self._hard_update(self.target_model, self.model)
            return

        if self.target_update == TargetUpdate.SOFT:
            self._soft_update(self.target_model, self.model)
            return

        raise KeyError(f"Target model update unknown: {self.target_update}")

    def _regularization_loss(
        self,
        rewards: th.Tensor,
        dones: th.Tensor,
        states: th.Tensor,
        actions: th.Tensor,
        next_states: th.Tensor,
        model: th.nn.Module | None = None,
    ) -> th.Tensor:
        if not self.regularizers:
            return states.new_tensor(0.0)

        if model is None:
            model = self.model

        loss = states.new_tensor(0.0)

        for regularizer, lam in zip(self.regularizers, self.reg_lams):
            value = regularizer(
                model,
                rewards,
                dones,
                states,
                actions,
                next_states,
            )

            if not th.is_tensor(value):
                value = states.new_tensor(float(value))

            loss = loss + lam * value

        return loss

    def optimize(
        self,
        loss: th.Tensor,
        optimizer: th.optim.Optimizer | None = None,
        parameters: list[th.nn.Parameter] | None = None,
    ):
        if optimizer is None:
            optimizer = self.optimizer

        if parameters is None:
            parameters = self.parameters

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if self.clip_grad:
            th.nn.utils.clip_grad_norm_(
                parameters,
                self.grad_norm_clip,
            )

        optimizer.step()

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