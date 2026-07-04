import torch as th

from .base import OffPolicyLearner
from .configs import SACConfig


class SACLearner(OffPolicyLearner):
    config: SACConfig

    def __init__(
        self,
        actor: th.nn.Module,
        critic1: th.nn.Module,
        critic2: th.nn.Module,
        actor_optimizer: th.optim.Optimizer,
        critic1_optimizer: th.optim.Optimizer,
        critic2_optimizer: th.optim.Optimizer,
        config: SACConfig,
    ):

        self.actor = actor.to(self.device)
        self.critic1 = critic1.to(self.device)
        self.critic2 = critic2.to(self.device)

        self.actor_optimizer = actor_optimizer
        self.critic1_optimizer = critic1_optimizer
        self.critic2_optimizer = critic2_optimizer

        self.target_entropy = config.target_entropy