import torch as th
from enum import Enum
from typing import Callable
from dataclasses import dataclass, field


class TargetUpdate(str, Enum):
    HARD = "hard"
    SOFT = "soft"


@dataclass
class OffPolicyConfig:
    device: th.device | str = "cpu"

    regularizers: list[Callable[..., th.Tensor | float]] = field(default_factory=list)
    reg_lams: list[float] = field(default_factory=list)

    clip_grad: bool = True
    grad_norm_clip: float = 1.0

    gamma: float = 0.99

    use_target_model: bool = True
    target_update: TargetUpdate = TargetUpdate.SOFT
    target_update_interval: int = 100
    soft_target_update_param: float = 0.1

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
    target_entropy: float | None = None