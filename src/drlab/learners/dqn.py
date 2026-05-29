import torch as th
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class DQNConfig:
    criterion: th.nn.Module = field(default_factory=th.nn.MSELoss)
    regularizers: list[Callable[..., th.Tensor | float]] = field(default_factory=list)
    reg_lams: list[float] = field(default_factory=list)
    clip_grad: bool = True
    grad_norm_clip: float = 1.0
    gamma: float = 0.99
    double_q: bool = True
    device: th.device | str = "cpu"
    use_target_model: bool = True
    target_update: str = "soft"
    target_update_interval: int = 100
    soft_target_update_param: float = 0.1
    num_actions: int = 2

class DQN:
    def __init__(
        self,
        model: th.nn.Module,
        optimizer: th.optim.Optimizer,
        config: DQNConfig,
    ):
        device = th.device(config.device)
        self.model = model
        self.model.to(device)
        self.gamma = config.gamma
        self.optimizer = optimizer
        self.criterion = config.criterion
        self.clip_grad = config.clip_grad
        self.grad_norm_clip = config.grad_norm_clip
        self.all_parameters = list(model.parameters())
        self.double_q = config.double_q
        self.device = device
        self.num_actions = config.num_actions
        self.regularizers = config.regularizers
        self.reg_lams = config.reg_lams

        self.target_update_calls = 0
        self.use_target_model = config.use_target_model
        self.soft_target_update_param = config.soft_target_update_param
        self.target_update = config.target_update
        self.target_update_interval = config.target_update_interval

        if len(self.regularizers) != len(self.reg_lams):
            raise ValueError("Length of regularizers and reg_lams lists must match.")

        if self.target_update not in {"hard", "soft"}:
            raise ValueError(f"target_update must be 'hard' or 'soft', got {self.target_update!r}")

        if self.target_update == "hard" and self.target_update_interval <= 0:
            raise ValueError(
                f"target_update_interval must be > 0 for hard updates, got {self.target_update_interval}"
            )
        
        if self.target_update == "soft" and not (0.0 < self.soft_target_update_param <= 1.0):
            raise ValueError("soft_target_update_param must be in (0, 1].")

        self.target_model = None
        if self.use_target_model:
            self.target_model = deepcopy(model)
            self.target_model.to(device)
            self.target_model.eval()
            for p in self.target_model.parameters():
                p.requires_grad = False

        if self.double_q and not self.use_target_model:
            raise ValueError("double_q=True requires use_target_model=True (needs a target network).")

    def target_model_update(self):
        if self.target_model is None:
            return

        if self.target_update == "hard":
            self.target_update_calls = (self.target_update_calls + 1) % self.target_update_interval
            if self.target_update_calls != 0:
                return
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_model.eval()

        elif self.target_update == "soft":
            tau = self.soft_target_update_param
            with th.no_grad():
                for tp, mp in zip(self.target_model.parameters(), self.model.parameters()):
                    tp.copy_(tp * (1 - tau) + mp * tau)

        else:
            raise KeyError(f"Target model update unknown: {self.target_update}")


    def q_values(self, states: th.Tensor, target: bool = False) -> th.Tensor:
        net = self.target_model if target else self.model
        return net(states)[:,:self.num_actions]

    def _current_values(self, states: th.Tensor, actions: th.Tensor):
        qvalues = self.q_values(states, target=False)
        return qvalues.gather(dim=-1, index=actions)

    def _next_values(self, next_states: th.Tensor) -> th.Tensor:
        with th.no_grad():
            if self.double_q:
                # action selection from online net, evaluation from target net
                online_q = self.q_values(next_states, target=False)
                actions = online_q.argmax(dim=-1, keepdim=True)
                target_q = self.q_values(next_states, target=True)
                return target_q.gather(dim=-1, index=actions)
            else:
                q = self.q_values(next_states, target=self.use_target_model)
                return q.max(dim=-1, keepdim=True)[0]

    def train(
        self,
        rewards: th.Tensor,      # float32, [B,1]
        dones: th.Tensor,        # bool or float(0/1), [B,1]
        states: th.Tensor,       # float32, [B, obs_dim] or [B,C,H,W]
        actions: th.Tensor,      # int64, [B,1]
        next_states: th.Tensor,  # float32, same as states
    ) -> float:
        
        self.model.train(True)

        # Compute targets and loss
        not_done = 1.0 - dones.float()
        targets = rewards + self.gamma * (not_done * self._next_values(next_states))
        pred = self._current_values(states, actions)
        td_loss = self.criterion(pred, targets) 
        reg_loss = (
            sum(
                lam * reg(self.model, rewards, dones, states, actions, next_states)
                for reg, lam in zip(self.regularizers, self.reg_lams)
            )
            if self.regularizers else 0.0
        )
        loss = td_loss + reg_loss

        # Optimize
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self.clip_grad:
            th.nn.utils.clip_grad_norm_(self.all_parameters, self.grad_norm_clip)
        self.optimizer.step()

        self.target_model_update()
        return float(loss.item())
