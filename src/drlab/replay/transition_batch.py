from __future__ import annotations
import numpy as np
import torch as th
from dataclasses import dataclass

@dataclass
class TransitionBatch:
    states: th.Tensor | np.ndarray       # [B, obs_dim]
    actions: th.Tensor | np.ndarray      # [B, 1] (int64/float32)
    rewards: th.Tensor | np.ndarray      # [B, 1] (float)
    terminated: th.Tensor | np.ndarray   # [B, 1] (bool), disables bootstrapping
    truncated: th.Tensor | np.ndarray    # [B, 1] (bool), ends an episode only
    next_states: th.Tensor | np.ndarray  # [B, obs_dim]
    returns: th.Tensor | np.ndarray      # [B, 1] (float)

    def to(self, device: th.device | str) -> TransitionBatch:
        d = th.device(device)

        def move(value: th.Tensor | np.ndarray) -> th.Tensor:
            if isinstance(value, np.ndarray):
                return th.from_numpy(value).to(d)
            return value.to(d)

        return TransitionBatch(
            states=move(self.states),
            actions=move(self.actions),
            rewards=move(self.rewards),
            terminated=move(self.terminated),
            truncated=move(self.truncated),
            next_states=move(self.next_states),
            returns=move(self.returns),
        )

    def cat(self, other: TransitionBatch) -> TransitionBatch:
        return TransitionBatch(
            states=th.cat([self.states, other.states], dim=0),
            actions=th.cat([self.actions, other.actions], dim=0),
            rewards=th.cat([self.rewards, other.rewards], dim=0),
            terminated=th.cat([self.terminated, other.terminated], dim=0),
            truncated=th.cat([self.truncated, other.truncated], dim=0),
            next_states=th.cat([self.next_states, other.next_states], dim=0),
            returns=th.cat([self.returns, other.returns], dim=0),
        )
