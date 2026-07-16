import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch as th

from drlab.controllers import GreedyController, WarmupController
from drlab.experiments import OffPolicyExperiment, OffPolicyExperimentConfig
from drlab.learners import OffPolicyConfig, OffPolicyLearner
from drlab.replay import TransitionBatch


class FixedLogits(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("logits", th.tensor([1.0, 0.0]))

    def forward(self, obs):
        return self.logits.unsqueeze(0).expand(obs.shape[0], -1)


class OneStepEnv(gym.Env):
    metadata = {}

    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.asarray([0.0], dtype=np.float32), {}

    def step(self, action):
        obs = np.asarray([0.0], dtype=np.float32)
        return obs, 0.0, True, False, {}


class CountingLearner(OffPolicyLearner):
    def __init__(self):
        super().__init__(OffPolicyConfig())
        self.train_calls = 0

    def train(
        self,
        rewards: th.Tensor,
        terminated: th.Tensor,
        states: th.Tensor,
        actions: th.Tensor,
        next_states: th.Tensor,
    ) -> float:
        self.train_calls += 1
        return 7.0


def one_transition() -> TransitionBatch:
    return TransitionBatch(
        states=np.asarray([[0.0]], dtype=np.float32),
        actions=np.asarray([[0]], dtype=np.int64),
        rewards=np.asarray([[1.0]], dtype=np.float32),
        terminated=np.asarray([[False]], dtype=np.bool_),
        truncated=np.asarray([[False]], dtype=np.bool_),
        next_states=np.asarray([[1.0]], dtype=np.float32),
        returns=np.asarray([[1.0]], dtype=np.float32),
    )


class OffPolicyExperimentTest(unittest.TestCase):
    def make_experiment(self, config: OffPolicyExperimentConfig) -> OffPolicyExperiment:
        env = OneStepEnv()
        learner = CountingLearner()
        controller = GreedyController(FixedLogits(), num_actions=2)
        return OffPolicyExperiment(env, controller, learner, config)

    def test_learning_starts_default_tracks_batch_size(self):
        with tempfile.TemporaryDirectory() as log_dir:
            experiment = self.make_experiment(
                OffPolicyExperimentConfig(
                    max_steps=1,
                    batch_size=256,
                    log_dir=log_dir,
                )
            )
            try:
                self.assertEqual(experiment.learning_starts, 256)
            finally:
                experiment.writer.close()

    def test_learning_starts_uses_current_steps_after_increment(self):
        with tempfile.TemporaryDirectory() as log_dir:
            experiment = self.make_experiment(
                OffPolicyExperimentConfig(
                    max_steps=2,
                    batch_size=2,
                    replay_buffer_size=4,
                    use_last_episode=False,
                    log_dir=log_dir,
                )
            )
            try:
                batch = one_transition()

                experiment.steps = 1
                loss = experiment._learn_from_batch(batch, None)
                self.assertEqual(loss, 0.0)
                self.assertEqual(experiment.learner.train_calls, 0)

                experiment.steps = 2
                loss = experiment._learn_from_batch(batch, None)
                self.assertEqual(loss, 7.0)
                self.assertEqual(experiment.learner.train_calls, 1)
            finally:
                experiment.writer.close()

    def test_warmup_steps_wraps_controller(self):
        with tempfile.TemporaryDirectory() as log_dir:
            experiment = self.make_experiment(
                OffPolicyExperimentConfig(
                    max_steps=1,
                    warmup_steps=10,
                    log_dir=log_dir,
                )
            )
            try:
                self.assertEqual(experiment.warmup_steps, 10)
                self.assertIsInstance(experiment.runner.controller, WarmupController)
                self.assertEqual(experiment.runner.controller.warmup_steps, 10)
            finally:
                experiment.writer.close()

    def test_zero_warmup_steps_keeps_original_controller(self):
        with tempfile.TemporaryDirectory() as log_dir:
            experiment = self.make_experiment(
                OffPolicyExperimentConfig(max_steps=1, log_dir=log_dir)
            )
            try:
                self.assertNotIsInstance(experiment.runner.controller, WarmupController)
            finally:
                experiment.writer.close()

    def test_negative_warmup_steps_is_rejected(self):
        with tempfile.TemporaryDirectory() as log_dir:
            with self.assertRaisesRegex(ValueError, "warmup_steps"):
                self.make_experiment(
                    OffPolicyExperimentConfig(
                        max_steps=1,
                        warmup_steps=-1,
                        log_dir=log_dir,
                    )
                )
