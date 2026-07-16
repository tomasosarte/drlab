import unittest

import gymnasium as gym
import numpy as np
import torch as th

from drlab.controllers import GreedyController
from drlab.replay import TransitionBatch
from drlab.runners import Runner


class FixedLogits(th.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", th.as_tensor(logits, dtype=th.float32))

    def forward(self, obs):
        return self.logits.unsqueeze(0).expand(obs.shape[0], -1)


class ThreeStepEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self.actions = []
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.actions = []
        return np.asarray([0.0], dtype=np.float32), {}

    def step(self, action):
        self.actions.append(int(action))
        self.t += 1
        reward = float(action + 1)
        terminated = self.t >= 3
        truncated = False
        obs = np.asarray([float(self.t)], dtype=np.float32)
        return obs, reward, terminated, truncated, {}


class TruncatingEnv(ThreeStepEnv):
    def step(self, action):
        obs, reward, _, _, info = super().step(action)
        return obs, reward, False, self.t >= 3, info


class RunnerTest(unittest.TestCase):
    def test_run_zero_collects_one_complete_episode_with_returns(self):
        env = ThreeStepEnv()
        model = FixedLogits([0.0, 1.0])
        controller = GreedyController(model, num_actions=2)
        runner = Runner(env, controller, calculate_returns=True, gamma=1.0)

        batch, ep_returns, ep_lengths, last_episode = runner.run(0)

        self.assertIsInstance(batch, TransitionBatch)
        self.assertIsInstance(last_episode, TransitionBatch)
        self.assertEqual(batch.states.shape, (3, 1))
        self.assertEqual(batch.actions.squeeze(-1).tolist(), [1, 1, 1])
        self.assertEqual(ep_returns, [6.0])
        self.assertEqual(ep_lengths, [3])
        self.assertTrue(th.equal(batch.returns.squeeze(-1), th.tensor([6.0, 4.0, 2.0])))
        self.assertTrue(th.equal(last_episode.returns.squeeze(-1), th.tensor([6.0, 4.0, 2.0])))
        self.assertTrue(batch.terminated[-1].item())
        self.assertFalse(batch.truncated.any().item())

    def test_truncation_ends_episode_without_marking_transition_terminal(self):
        env = TruncatingEnv()
        model = FixedLogits([0.0, 1.0])
        controller = GreedyController(model, num_actions=2)
        runner = Runner(env, controller, calculate_returns=False)

        batch, ep_returns, ep_lengths, last_episode = runner.run(0)

        self.assertEqual(ep_returns, [6.0])
        self.assertEqual(ep_lengths, [3])
        self.assertIsNotNone(last_episode)
        self.assertFalse(batch.terminated.any().item())
        self.assertTrue(batch.truncated[-1].item())
        self.assertTrue((batch.terminated | batch.truncated)[-1].item())

    def test_run_positive_steps_can_return_partial_batch(self):
        env = ThreeStepEnv()
        model = FixedLogits([1.0, 0.0])
        controller = GreedyController(model, num_actions=2)
        runner = Runner(env, controller, calculate_returns=False)

        batch, ep_returns, ep_lengths, last_episode = runner.run(2)

        self.assertEqual(batch.states.shape, (2, 1))
        self.assertEqual(batch.actions.squeeze(-1).tolist(), [0, 0])
        self.assertEqual(ep_returns, [])
        self.assertEqual(ep_lengths, [])
        self.assertIsNone(last_episode)

    def test_run_can_return_numpy_batch(self):
        env = ThreeStepEnv()
        model = FixedLogits([0.0, 1.0])
        controller = GreedyController(model, num_actions=2)
        runner = Runner(env, controller, calculate_returns=True, gamma=1.0)

        batch, ep_returns, ep_lengths, last_episode = runner.run(0, as_numpy=True)

        self.assertIsInstance(batch, TransitionBatch)
        self.assertIsInstance(last_episode, TransitionBatch)
        self.assertIsInstance(batch.states, np.ndarray)
        self.assertEqual(batch.states.dtype, np.float32)
        self.assertEqual(batch.actions.dtype, np.int64)
        self.assertEqual(batch.actions.squeeze(-1).tolist(), [1, 1, 1])
        self.assertEqual(ep_returns, [6.0])
        self.assertEqual(ep_lengths, [3])
        np.testing.assert_array_equal(
            batch.returns.squeeze(-1),
            np.asarray([6.0, 4.0, 2.0], dtype=np.float32),
        )
