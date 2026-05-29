from __future__ import annotations
import numpy as np
import torch as th
from typing import Tuple

from drlab.replay.transition_batch import TransitionBatch

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        device: th.device | str = "cpu",
    ):
        self.capacity = int(capacity)
        self.device = th.device(device)
        self.obs_shape = obs_shape

        self.states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, 1), dtype=np.int64)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.bool_)
        self.returns = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        states: np.ndarray,       # [B, *obs_shape]
        actions: np.ndarray,      # [B] or [B,1]
        rewards: np.ndarray,      # [B] or [B,1]
        dones: np.ndarray,        # [B] or [B,1] (bool)
        next_states: np.ndarray,  # [B, *obs_shape]
        returns: np.ndarray       # [B, 1]

    ) -> None:
        B = states.shape[0]
        assert states.shape[1:] == self.obs_shape
        assert next_states.shape[1:] == self.obs_shape
        assert B <= self.capacity, f"Batch size B={B} must be <= capacity={self.capacity}"

        actions = np.asarray(actions).reshape(B, 1).astype(np.int64)
        rewards = np.asarray(rewards).reshape(B, 1).astype(np.float32)
        dones   = np.asarray(dones).reshape(B, 1).astype(np.bool_)
        returns = np.asarray(returns).reshape(B, 1).astype(np.float32)

        idx = (self.ptr + np.arange(B)) % self.capacity

        self.states[idx] = states
        self.next_states[idx] = next_states
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.dones[idx] = dones
        self.returns[idx] = returns

        self.ptr = (self.ptr + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    def get_all(self) -> TransitionBatch:
        states = th.from_numpy(self.states[:self.size]).to(self.device)
        next_states = th.from_numpy(self.next_states[:self.size]).to(self.device)
        actions = th.from_numpy(self.actions[:self.size]).to(self.device)
        rewards = th.from_numpy(self.rewards[:self.size]).to(self.device)
        dones = th.from_numpy(self.dones[:self.size]).to(self.device)
        returns = th.from_numpy(self.returns[:self.size]).to(self.device)
        return TransitionBatch(states, actions, rewards, dones, next_states, returns)

    def sample(self, batch_size: int) -> TransitionBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        b = min(int(batch_size), self.size)
        idx = np.random.randint(0, self.size, size=b)

        states = th.from_numpy(self.states[idx]).to(self.device)
        next_states = th.from_numpy(self.next_states[idx]).to(self.device)
        actions = th.from_numpy(self.actions[idx]).to(self.device)               # long already
        rewards = th.from_numpy(self.rewards[idx]).to(self.device)               # float32
        dones = th.from_numpy(self.dones[idx]).to(self.device)                   # bool
        returns = th.from_numpy(self.returns[idx]).to(self.device)               # float32

        return TransitionBatch(states, actions, rewards, dones, next_states, returns)
