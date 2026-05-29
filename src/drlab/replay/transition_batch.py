from __future__ import annotations
import torch as th
from dataclasses import dataclass

@dataclass
class TransitionBatch:
    states: th.Tensor       # [B, obs_dim]
    actions: th.Tensor      # [B, 1] (long)
    rewards: th.Tensor      # [B, 1] (float)
    dones: th.Tensor        # [B, 1] (bool)
    next_states: th.Tensor  # [B, obs_dim]
    returns: th.Tensor      # [B, 1] (float)

    def to(self, device: th.device | str) -> TransitionBatch:
        d = th.device(device)
        return TransitionBatch(
            states=self.states.to(d),
            actions=self.actions.to(d),
            rewards=self.rewards.to(d),
            dones=self.dones.to(d),
            next_states=self.next_states.to(d),
            returns=self.returns.to(d),
        )

    def cat(self, other: TransitionBatch) -> TransitionBatch:
        return TransitionBatch(
            states=th.cat([self.states, other.states], dim=0),
            actions=th.cat([self.actions, other.actions], dim=0),
            rewards=th.cat([self.rewards, other.rewards], dim=0),
            dones=th.cat([self.dones, other.dones], dim=0),
            next_states=th.cat([self.next_states, other.next_states], dim=0),
            returns=th.cat([self.returns, other.returns], dim=0),
        )