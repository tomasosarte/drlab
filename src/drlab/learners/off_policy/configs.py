from dataclasses import dataclass, field
from enum import Enum
from math import prod
from typing import Callable, Tuple

import torch as th


class TargetUpdate(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass
class OffPolicyConfig:
    device: th.device | str = "cpu"

    gamma: float = 0.99

    regularizers: list[Callable[..., th.Tensor | float]] = field(default_factory=list)
    reg_lams: list[float] = field(default_factory=list)

    clip_grad: bool = True
    grad_norm_clip: float = 1.0

    use_target_model: bool = True
    target_update: TargetUpdate | str = TargetUpdate.SOFT
    target_update_interval: int = 100
    soft_target_update_param: float = 0.005

    action_shape: Tuple[int, ...] = (1,)

    def __post_init__(self):
        if isinstance(self.target_update, str):
            self.target_update = TargetUpdate(self.target_update)


@dataclass
class DQNConfig(OffPolicyConfig):
    criterion: th.nn.Module = field(default_factory=th.nn.MSELoss)
    double_q: bool = True
    num_actions: int = 2


@dataclass
class SACConfig(OffPolicyConfig):
    criterion: th.nn.Module = field(default_factory=th.nn.MSELoss)

    target_entropy: float | None = None
    alpha_lr: float = 3e-4

    min_log_std: float = -20.0
    max_log_std: float = 2.0

    # SAC normally uses soft target updates with small tau.
    soft_target_update_param: float = 0.005

    @property
    def action_dim(self) -> int:
        return int(prod(self.action_shape))

    def __post_init__(self):
        super().__post_init__()

        if self.action_dim <= 0:
            raise ValueError("action_dim must be > 0.")

        if self.alpha_lr <= 0.0:
            raise ValueError("alpha_lr must be > 0.")

        if self.min_log_std >= self.max_log_std:
            raise ValueError("min_log_std must be < max_log_std.")