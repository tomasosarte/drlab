import torch as th
from enum import Enum
from typing import Callable
from dataclasses import dataclass, field


class ValueTargets(str, Enum):
    TD = "td"
    RETURNS = "returns"

@dataclass
class OnPolicyConfig:
    device: th.device | str = "cpu"

    regularizers: list[Callable[..., th.Tensor | float]] = field(default_factory=list)
    reg_lams: list[float] = field(default_factory=list)

    num_actions: int = 2

    clip_grad: bool = True
    grad_norm_clip: float = 1.0

    use_entropy: bool = False
    entropy_max_lambda: float = 0.0
    entropy_min_lambda: float = 0.0
    entropy_anneal_steps: int = 1

@dataclass
class ReinforceConfig(OnPolicyConfig):
    normalize_returns: bool = False

@dataclass
class ActorCriticConfig(ReinforceConfig):
    use_bias: bool = True

    value_criterion: Callable = field(default_factory=th.nn.MSELoss)
    value_lambda: float = 0.1
    value_targets: ValueTargets = ValueTargets.TD

    gamma: float = 0.99
    advantage_bootstrap: bool = True
    normalize_advantages: bool = False

    def __post_init__(self):
        if isinstance(self.value_targets, str):
            self.value_targets = ValueTargets(self.value_targets)

@dataclass
class PPOConfig(ActorCriticConfig):
    ppo_clipping: float = 0.1
    ppo_iterations: int = 4
