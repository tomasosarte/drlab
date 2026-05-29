import torch as th
import numpy as np
import gymnasium as gym

from drlab.replay import TransitionBatch
from drlab.controllers import Controller

class Runner:

    def __init__(
            self,
            env: gym.Env,
            controller: Controller,
            calculate_returns: bool = True,
            return_last_episode: bool = True,
            gamma: float = 0.99,
            device: str = "cpu"
        ):

        self.env = env
        self.controller = controller
        self.gamma = gamma
        self.device = device
        self.state, _ = self.env.reset()
        self.calculate_returns = calculate_returns
        self.return_last_episode = return_last_episode
        self.ep_len, self.ep_ret = 0, 0.0
        self._ep_states, self._ep_actions, self._ep_rewards = [], [], []
        self._ep_dones, self._ep_next_states, self._ep_returns = [], [], []

    def _discounted_returns(self, rewards: list[float]) -> list[float]:
        out = [0.0] * len(rewards)
        running = 0.0
        
        for i in range(len(rewards) - 1, -1, -1):
            running = rewards[i] + self.gamma * running
            out[i] = running
        
        return out

    def run(
            self,
            num_steps: int
        ) -> tuple[TransitionBatch, list, list, TransitionBatch]:

        self.controller.model.eval()
        last_episode = None
        ep_returns, ep_lengths = [], []
        ep_start = 0
        states, actions, rewards, dones, next_states, returns = [], [], [], [], [], []
        run_one_episode = num_steps <= 0
        
        st = 0
        while True:

            # 1. Environment step
            state_t = th.as_tensor(self.state, dtype=th.float32, device=self.device).unsqueeze(0)
            with th.inference_mode(): 
                action = self.controller.choose(state_t).item()
            next_state, reward, terminated, truncated, _ = self.env.step(action) 
            done = terminated or truncated

            # 2. Store transition
            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            returns.append(0)
            self.ep_ret += reward
            self.ep_len += 1
            if self.return_last_episode:
                self._ep_states.append(self.state)
                self._ep_actions.append(action)
                self._ep_rewards.append(reward)
                self._ep_dones.append(done)
                self._ep_next_states.append(next_state)
                self._ep_returns.append(0)

            # 3. Compute cumlative returns
            if self.calculate_returns and (done or st == num_steps - 1):
                ep_rewards = rewards[ep_start: st + 1]
                discounted_returns = self._discounted_returns(ep_rewards)

                returns[ep_start: st + 1] = discounted_returns
                if self.return_last_episode and done:
                    self._ep_returns = discounted_returns

            # 4. Update state
            self.state = next_state
            if done:
                if self.return_last_episode:
                    last_episode = TransitionBatch(
                        states=th.from_numpy(np.asarray(self._ep_states, dtype=np.float32)),
                        actions=th.as_tensor(self._ep_actions, dtype=th.int64).unsqueeze(-1),
                        rewards=th.as_tensor(self._ep_rewards, dtype=th.float32).unsqueeze(-1),
                        dones=th.as_tensor(self._ep_dones, dtype=th.bool).unsqueeze(-1),
                        next_states=th.from_numpy(np.asarray(self._ep_next_states, dtype=np.float32)),
                        returns=th.as_tensor(self._ep_returns, dtype=th.float32).unsqueeze(-1),
                    )
                    self._ep_states, self._ep_actions, self._ep_rewards = [], [], []
                    self._ep_dones, self._ep_next_states, self._ep_returns = [], [], []
                ep_start = st + 1
                ep_returns.append(self.ep_ret)
                ep_lengths.append(self.ep_len)
                self.ep_ret, self.ep_len = 0.0, 0
                self.state, _ = self.env.reset()
                if run_one_episode:
                    break
            
            st += 1
            if st >= num_steps and not run_one_episode:
                break
            
        batch = TransitionBatch(
            states=th.from_numpy(np.asarray(states, dtype=np.float32)),
            actions=th.as_tensor(actions, dtype=th.int64).unsqueeze(-1),
            rewards=th.as_tensor(rewards, dtype=th.float32).unsqueeze(-1),
            dones=th.as_tensor(dones, dtype=th.bool).unsqueeze(-1),
            next_states=th.from_numpy(np.asarray(next_states, dtype=np.float32)),
            returns=th.as_tensor(returns, dtype=th.float32).unsqueeze(-1)
        )
        return batch, ep_returns, ep_lengths, last_episode
