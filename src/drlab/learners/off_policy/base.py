import torch as th
from copy import deepcopy
from abc import ABC, abstractmethod

from .configs import OffPolicyConfig, TargetUpdate


class OffPolicyLearner(ABC):
    def __init__(self, config: OffPolicyConfig):
        self.config = config
        self.device = th.device(config.device)

        self.gamma = config.gamma

        self.regularizers = config.regularizers
        self.reg_lams = config.reg_lams

        self.clip_grad = config.clip_grad
        self.grad_norm_clip = config.grad_norm_clip

        self.use_target_model = config.use_target_model
        self.target_update = TargetUpdate(config.target_update)
        self.target_update_interval = config.target_update_interval
        self.soft_target_update_param = config.soft_target_update_param
        self.target_update_calls = 0

        self.last_losses: dict[str, float] = {}

        self._validate_config()

    def _validate_config(self) -> None:
        if len(self.regularizers) != len(self.reg_lams):
            raise ValueError("regularizers and reg_lams must have same length.")

        if self.target_update not in {TargetUpdate.HARD, TargetUpdate.SOFT}:
            raise ValueError(f"Invalid target_update: {self.target_update}")

        if self.target_update == TargetUpdate.HARD and self.target_update_interval <= 0:
            raise ValueError("target_update_interval must be > 0 for hard updates.")

        if self.target_update == TargetUpdate.SOFT:
            tau = self.soft_target_update_param
            if not 0.0 < tau <= 1.0:
                raise ValueError("soft_target_update_param must be in (0, 1].")

    def make_target_model(self, model: th.nn.Module) -> th.nn.Module:
        target = deepcopy(model).to(self.device)
        target.eval()

        for p in target.parameters():
            p.requires_grad = False

        return target

    def hard_update(self, target: th.nn.Module, source: th.nn.Module):
        target.load_state_dict(source.state_dict())
        target.eval()

    def soft_update(self, target: th.nn.Module, source: th.nn.Module):
        tau = self.soft_target_update_param

        with th.no_grad():
            for target_p, source_p in zip(target.parameters(), source.parameters()):
                target_p.data.mul_(1.0 - tau)
                target_p.data.add_(tau * source_p.data)

    def update_targets(
        self,
        pairs: list[tuple[th.nn.Module, th.nn.Module]],
    ):
        if self.target_update == TargetUpdate.SOFT:
            for target, source in pairs:
                self.soft_update(target, source)
            return

        if self.target_update == TargetUpdate.HARD:
            self.target_update_calls += 1

            if self.target_update_calls % self.target_update_interval == 0:
                for target, source in pairs:
                    self.hard_update(target, source)

            return

        raise ValueError(f"Unknown target update: {self.target_update}")

    def regularization_loss(
        self,
        model: th.nn.Module,
        rewards: th.Tensor,
        terminated: th.Tensor,
        states: th.Tensor,
        actions: th.Tensor,
        next_states: th.Tensor,
    ) -> th.Tensor:
        if not self.regularizers:
            return states.new_tensor(0.0)

        loss = states.new_tensor(0.0)

        for regularizer, lam in zip(self.regularizers, self.reg_lams):
            value = regularizer(
                model,
                rewards,
                terminated,
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
        optimizer: th.optim.Optimizer,
        parameters: list[th.nn.Parameter]
    ):
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

    @abstractmethod
    def train(
        self,
        rewards: th.Tensor,
        terminated: th.Tensor,
        states: th.Tensor,
        actions: th.Tensor,
        next_states: th.Tensor,
    ) -> float:
        pass
