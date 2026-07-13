from typing import Any

import numpy as np
import torch as th

from .base import ContinuousActionController, Controller, DiscreteActionController


class WarmupController(ContinuousActionController, DiscreteActionController):
    def __init__(self, controller: Controller, action_space: Any, warmup_steps: int):
        self.controller = controller
        self.action_space = action_space
        self.warmup_steps = warmup_steps
        self.steps = 0
        self.model = controller.model

        if hasattr(controller, "num_actions"):
            self.num_actions = controller.num_actions
        if hasattr(controller, "action_dim"):
            self.action_dim = controller.action_dim

        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0.")

    def choose(
        self,
        obs: th.Tensor,
        **kwargs,
    ) -> th.Tensor:
        if self.steps >= self.warmup_steps:
            return self.controller.choose(obs, **kwargs)

        batch_size = obs.shape[0]
        self.steps += batch_size
        actions = [self.action_space.sample() for _ in range(batch_size)]
        return th.as_tensor(np.asarray(actions), device=obs.device)

    def probabilities(self, obs: th.Tensor, **kwargs) -> th.Tensor:
        if self.steps >= self.warmup_steps:
            return self.controller.probabilities(obs, **kwargs)

        return th.ones((obs.shape[0], self.num_actions), device=obs.device) / self.num_actions
