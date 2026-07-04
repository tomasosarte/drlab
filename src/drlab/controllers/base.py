from __future__ import annotations

import torch as th
from abc import ABC, abstractmethod


class Controller(ABC):
    """Base controller interface for all action spaces."""

    model: th.nn.Module

    @abstractmethod
    def choose(self, obs: th.Tensor, **kwargs) -> th.Tensor: ...


class DiscreteActionController(Controller):
    """Controller interface for discrete action spaces."""

    num_actions: int

    @abstractmethod
    def probabilities(self, obs: th.Tensor, **kwargs) -> th.Tensor: ...


class ContinuousActionController(Controller):
    """Controller interface for continuous action spaces."""

    action_dim: int
