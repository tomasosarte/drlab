import torch as th
from .base import ContinuousActionController


class GaussianController(ContinuousActionController):

    def __init__(
        self,
        model: th.nn.Module,
        action_dim: int,
        deterministic: bool = False,
    ):
        self.model = model
        self.action_dim = action_dim
        self.num_actions = action_dim
        self.deterministic = deterministic

    def choose(self, obs: th.Tensor, **kwargs) -> th.Tensor:
        output = self.model(obs)

        mean = output[:, :self.action_dim]
        log_std = output[:, self.action_dim: 2 * self.action_dim]
        log_std = th.clamp(log_std, -20, 2)

        if self.deterministic:
            u = mean
        else:
            std = log_std.exp()
            eps = th.randn_like(mean)
            u = mean + std * eps

        return th.tanh(u)
