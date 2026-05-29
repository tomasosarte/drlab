import numpy as np
import torch as th
from .base import Controller

class EpsilonGreedyController(Controller):

    def __init__(
        self,
        controller: Controller,
        num_actions: int,
        max_eps: float = 1.0,
        min_eps: float = 0.1,
        anneal_steps: int = 10_000,
    ):
        self.controller = controller
        self.num_actions = num_actions
        self.model = controller.model
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.anneal_steps = anneal_steps
        self.num_decisions = 0

        if anneal_steps <= 1:
            raise ValueError("anneal_steps must be >= 2")

    def epsilon(self) -> float:
        frac = max(1 - self.num_decisions / (self.anneal_steps - 1), 0.0)
        return frac * (self.max_eps - self.min_eps) + self.min_eps

    def choose(self, obs: th.Tensor, increase_counter: bool = True, **kwargs) -> th.Tensor:
        eps = self.epsilon()
        if increase_counter:
            self.num_decisions += 1

        B = obs.shape[0] if obs.ndim > 1 else 1
        if np.random.rand() < eps:
            return th.randint(self.num_actions, (B,), device=obs.device, dtype=th.long)

        return self.controller.choose(obs, **kwargs)

    def probabilities(self, obs: th.Tensor, **kwargs) -> th.Tensor:
        eps = self.epsilon()
        greedy = self.controller.probabilities(obs, **kwargs)  # one-hot on argmax, shape [B,A]
        B = greedy.shape[0]
        uniform = th.full((B, self.num_actions), eps / self.num_actions, device=greedy.device)
        return uniform + (1 - eps) * greedy
