import torch as th
import numpy as np
import gymnasium as gym

from drlab.replay import TransitionBatch
from drlab.controllers import Controller, ContinuousActionController

class Runner:

    def __init__(
            self,
            env: gym.Env,
            controller: Controller,
            calculate_returns: bool = True,
            return_last_episode: bool = True,
            gamma: float = 0.99,
            device: str = "cpu",
            continuous_actions: bool = False
        ):

        self.env = env
        self.controller = controller
        self.gamma = gamma
        self.device = device
        self.continuous_actions = continuous_actions
        
        if continuous_actions and not isinstance(self.controller, ContinuousActionController):
            raise TypeError(
            "continuous_actions=True requires controller to be a ContinuousActionController."
        )

        self.state, _ = self.env.reset()
        self.calculate_returns = calculate_returns
        self.return_last_episode = return_last_episode
        self.ep_len, self.ep_ret = 0, 0.0
        self._ep_states, self._ep_actions, self._ep_rewards = [], [], []
        self._ep_terminated, self._ep_truncated = [], []
        self._ep_next_states, self._ep_returns = [], []

    def _discounted_returns(self, rewards: list[float]) -> list[float]:
        out = [0.0] * len(rewards)
        running = 0.0
        
        for i in range(len(rewards) - 1, -1, -1):
            running = rewards[i] + self.gamma * running
            out[i] = running
        
        return out

    def _action_to_env(self, action_t: th.Tensor):
        if self.continuous_actions:
            action = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
            return np.clip(
                action,
                self.env.action_space.low,
                self.env.action_space.high,
            )

        return int(action_t.item())

    def _actions_to_array(self, actions) -> np.ndarray:
        if self.continuous_actions:
            return np.asarray(actions, dtype=np.float32)

        return np.asarray(actions, dtype=np.int64).reshape(-1, 1)

    def _make_batch(
        self,
        states,
        actions,
        rewards,
        terminated,
        truncated,
        next_states,
        returns,
        as_numpy: bool,
    ) -> TransitionBatch:
        states = np.asarray(states, dtype=np.float32)
        actions = self._actions_to_array(actions)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        terminated = np.asarray(terminated, dtype=np.bool_).reshape(-1, 1)
        truncated = np.asarray(truncated, dtype=np.bool_).reshape(-1, 1)
        next_states = np.asarray(next_states, dtype=np.float32)
        returns = np.asarray(returns, dtype=np.float32).reshape(-1, 1)

        if as_numpy:
            return TransitionBatch(
                states, actions, rewards, terminated, truncated, next_states, returns
            )

        return TransitionBatch(
            states=th.from_numpy(states),
            actions=th.from_numpy(actions),
            rewards=th.from_numpy(rewards),
            terminated=th.from_numpy(terminated),
            truncated=th.from_numpy(truncated),
            next_states=th.from_numpy(next_states),
            returns=th.from_numpy(returns),
        )
    
    def run(
            self,
            num_steps: int,
            as_numpy: bool = False,
        ) -> tuple[TransitionBatch, list, list, TransitionBatch | None]:

        self.controller.model.eval()
        last_episode = None
        ep_returns, ep_lengths = [], []
        ep_start = 0
        states, actions, rewards = [], [], []
        terminated_flags, truncated_flags = [], []
        next_states, returns = [], []
        run_one_episode = num_steps <= 0
        
        st = 0
        while True:

            # 1. Environment step
            state_t = th.as_tensor(self.state, dtype=th.float32, device=self.device).unsqueeze(0)
            with th.inference_mode():
                action_t = self.controller.choose(state_t)
            action = self._action_to_env(action_t)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode_end = terminated or truncated

            # 2. Store transition
            states.append(self.state)
            actions.append(action)
            rewards.append(reward)
            terminated_flags.append(terminated)
            truncated_flags.append(truncated)
            next_states.append(next_state)
            returns.append(0)
            self.ep_ret += reward
            self.ep_len += 1
            if self.return_last_episode:
                self._ep_states.append(self.state)
                self._ep_actions.append(action)
                self._ep_rewards.append(reward)
                self._ep_terminated.append(terminated)
                self._ep_truncated.append(truncated)
                self._ep_next_states.append(next_state)
                self._ep_returns.append(0)

            # 3. Compute cumlative returns
            if self.calculate_returns and (episode_end or st == num_steps - 1):
                ep_rewards = rewards[ep_start: st + 1]
                discounted_returns = self._discounted_returns(ep_rewards)

                returns[ep_start: st + 1] = discounted_returns
                if self.return_last_episode and episode_end:
                    self._ep_returns = discounted_returns

            # 4. Update state
            self.state = next_state
            if episode_end:
                if self.return_last_episode:
                    last_episode = self._make_batch(
                        self._ep_states,
                        self._ep_actions,
                        self._ep_rewards,
                        self._ep_terminated,
                        self._ep_truncated,
                        self._ep_next_states,
                        self._ep_returns,
                        as_numpy,
                    )
                    self._ep_states, self._ep_actions, self._ep_rewards = [], [], []
                    self._ep_terminated, self._ep_truncated = [], []
                    self._ep_next_states, self._ep_returns = [], []
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
        
        batch = self._make_batch(
            states,
            actions,
            rewards,
            terminated_flags,
            truncated_flags,
            next_states,
            returns,
            as_numpy,
        )
        return batch, ep_returns, ep_lengths, last_episode
