import torch as th

from .configs import OnPolicyConfig

class OnPolicyLearner:

    def __init__(
        self,
        actor: th.nn.Module,
        optimizer: th.optim.Optimizer,
        config: OnPolicyConfig,
        ):

        self.actor = actor.to(config.device)
        self.optimizer = optimizer
        self.config = config
        self.device = th.device(config.device)
        self.parameters = list(self.actor.parameters())
        self.entropy_step = 0

        self._validate_config()

    def _validate_config(self):
        if len(self.config.regularizers) != len(self.config.reg_lams):
            raise ValueError("regularizers and reg_lams must have same length.")

        if self.config.use_entropy and self.config.entropy_anneal_steps <= 1:
            raise ValueError("entropy_anneal_steps must be > 1 when use_entropy=True.")

        if self.config.use_entropy and self.config.entropy_max_lambda < self.config.entropy_min_lambda:
            raise ValueError("entropy_max_lambda must be >= entropy_min_lambda.")
        
    def _entropy_lambda(self) -> float:
        progress = min(self.entropy_step / (self.config.entropy_anneal_steps - 1), 1.0)

        return (
            self.config.entropy_max_lambda
            + progress * (self.config.entropy_min_lambda - self.config.entropy_max_lambda)
        )
    
    def _entropy_loss(self, logits: th.Tensor) -> th.Tensor:
        if not self.config.use_entropy:
            return logits.new_tensor(0.0)
        entropy_lambda = self._entropy_lambda()
        probs = th.nn.functional.softmax(logits, dim=-1)
        log_probs = th.nn.functional.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        return -entropy_lambda * entropy
    
    def _regularization_loss(
        self,
        rewards: th.Tensor,
        dones: th.Tensor,
        states: th.Tensor,
        actions: th.Tensor,
        next_states: th.Tensor,
    ):
        if not self.config.regularizers:
            return 0.0

        return sum(
            lam * reg(self.actor, rewards, dones, states, actions, next_states)
            for reg, lam in zip(self.config.regularizers, self.config.reg_lams)
        )

    def _value_loss(
        self,
        returns: th.Tensor,
        rewards: th.Tensor,
        dones: th.Tensor,
        values: th.Tensor,
        next_values: th.Tensor,
    ) -> th.Tensor:
        raise NotImplementedError
    
    def _policy_loss(self, pi: th.Tensor, advantage: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    def _get_policy(self, logits: th.Tensor, actions: th.Tensor) -> th.Tensor:
        probs = th.nn.functional.softmax(logits, dim=-1)
        pi = probs.gather(dim=-1, index=actions)
        return pi

    def optimize(self, loss: th.Tensor):
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.config.clip_grad:
            th.nn.utils.clip_grad_norm_(
                self.parameters,
                self.config.grad_norm_clip,
            )
        self.optimizer.step()

    def forward(self, states: th.Tensor, get_values: bool = False):
        output = self.actor(states)
        logits = output[:, :self.config.num_actions]
        values = None
        if get_values:
            values = output[:, self.config.num_actions:self.config.num_actions + 1]
        return logits, values

    def requires_returns(self) -> bool:
        return True

    def _advantages(
            self, 
            returns: th.Tensor, 
            rewards: th.Tensor,
            dones: th.Tensor,
            values: th.Tensor,
            next_values: th.Tensor
        ) -> th.Tensor:
        raise NotImplementedError
    
    def train(
        self,
        rewards: th.Tensor,      # float32, [B,1]
        dones: th.Tensor,        # bool or float(0/1), [B,1]
        states: th.Tensor,       # float32, [B, obs_dim] or [B,C,H,W]
        actions: th.Tensor,      # int64, [B,1]
        next_states: th.Tensor,  # float32, same as states
        returns: th.Tensor,      # float32, [B,1]
    ) -> float:
        raise NotImplementedError  
