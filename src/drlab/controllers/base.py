from __future__ import annotations

import torch as th
from abc import ABC, abstractmethod

class Controller(ABC):
    """Abstract controller interface."""
    
    num_actions: int
    model: th.nn.Module = None  
    controller: Controller = None

    @abstractmethod
    def choose(self, obs: th.Tensor, **kwargs) -> th.Tensor: ...

    @abstractmethod
    def probabilities(self, obs: th.Tensor, **kwargs) -> th.Tensor: ...