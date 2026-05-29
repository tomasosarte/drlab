import torch as th
from .base import Controller

class StochasticController(Controller):

    def __init__(self, model: th.nn.Module, num_actions: int):
        
        self.num_actions = num_actions
        self.model = model

    def choose(self, obs: th.Tensor) -> th.Tensor:
        probs = self.probabilities(obs)
        return th.distributions.Categorical(probs=probs).sample()
        
    def probabilities(self, obs: th.Tensor) -> th.Tensor:
        output = self.model(obs)[:, :self.num_actions]
        return th.nn.functional.softmax(output, dim=-1)