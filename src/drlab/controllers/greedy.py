import torch as th
from .base import DiscreteActionController


class GreedyController(DiscreteActionController):

    def __init__(self, model: th.nn.Module, num_actions: int):
        self.num_actions = num_actions
        self.model = model

    def choose(self, obs: th.Tensor):
        output: th.Tensor = self.model(obs)[:, :self.num_actions]
        return th.argmax(output, dim=-1)

    def probabilities(self, obs: th.Tensor):
        output: th.Tensor = self.model(obs)[:, :self.num_actions]
        idx = output.argmax(dim=-1, keepdim=True)
        probs = th.zeros(output.shape[0], self.num_actions, device=output.device)
        probs.scatter_(dim=-1, index=idx, value=1.0)
        return probs
