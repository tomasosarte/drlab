import unittest

import numpy as np
import torch as th

from drlab.controllers import (
    ContinuousActionController,
    DiscreteActionController,
    EpsilonGreedyController,
    GaussianController,
    GreedyController,
    StochasticController,
    WarmupController,
)


class FixedLogits(th.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", th.as_tensor(logits, dtype=th.float32))

    def forward(self, obs):
        return self.logits.unsqueeze(0).expand(obs.shape[0], -1)


class FixedGaussianOutput(FixedLogits):
    pass


class FixedActionSpace:
    def __init__(self, action):
        self.action = action

    def sample(self):
        return self.action


class ControllerTest(unittest.TestCase):
    def test_discrete_controllers_share_discrete_base(self):
        model = FixedLogits([0.1, 2.0, -1.0])
        greedy = GreedyController(model, num_actions=3)
        epsilon_greedy = EpsilonGreedyController(
            greedy,
            num_actions=3,
            max_eps=0.0,
            min_eps=0.0,
            anneal_steps=2,
        )
        stochastic = StochasticController(model, num_actions=3)

        self.assertIsInstance(greedy, DiscreteActionController)
        self.assertIsInstance(epsilon_greedy, DiscreteActionController)
        self.assertIsInstance(stochastic, DiscreteActionController)

    def test_gaussian_controller_uses_continuous_base(self):
        model = FixedGaussianOutput([0.0, 1.0, -3.0, -3.0])
        controller = GaussianController(model, action_dim=2, deterministic=True)
        obs = th.zeros(3, 2)

        actions = controller.choose(obs)

        self.assertIsInstance(controller, ContinuousActionController)
        self.assertNotIsInstance(controller, DiscreteActionController)
        self.assertEqual(actions.shape, (3, 2))
        self.assertTrue(th.all(actions <= 1.0))
        self.assertTrue(th.all(actions >= -1.0))
        self.assertFalse(hasattr(controller, "probabilities"))

    def test_greedy_controller_selects_argmax_and_one_hot_probabilities(self):
        model = FixedLogits([0.1, 2.0, -1.0])
        controller = GreedyController(model, num_actions=3)
        obs = th.zeros(4, 2)

        actions = controller.choose(obs)
        probs = controller.probabilities(obs)

        self.assertTrue(th.equal(actions, th.full((4,), 1)))
        self.assertEqual(probs.shape, (4, 3))
        self.assertTrue(th.equal(probs[0], th.tensor([0.0, 1.0, 0.0])))

    def test_epsilon_greedy_can_act_greedily_without_exploration(self):
        model = FixedLogits([0.1, 2.0, -1.0])
        greedy = GreedyController(model, num_actions=3)
        controller = EpsilonGreedyController(
            greedy,
            num_actions=3,
            max_eps=0.0,
            min_eps=0.0,
            anneal_steps=2,
        )
        obs = th.zeros(2, 2)

        actions = controller.choose(obs)
        probs = controller.probabilities(obs)

        self.assertTrue(th.equal(actions, th.full((2,), 1)))
        self.assertTrue(th.equal(probs[0], th.tensor([0.0, 1.0, 0.0])))
        self.assertEqual(controller.num_decisions, 1)

    def test_epsilon_greedy_probabilities_include_uniform_exploration(self):
        model = FixedLogits([0.1, 2.0, -1.0])
        greedy = GreedyController(model, num_actions=3)
        controller = EpsilonGreedyController(
            greedy,
            num_actions=3,
            max_eps=0.6,
            min_eps=0.6,
            anneal_steps=2,
        )

        probs = controller.probabilities(th.zeros(1, 2))

        self.assertTrue(th.allclose(probs, th.tensor([[0.2, 0.6, 0.2]])))

    def test_stochastic_controller_returns_softmax_probabilities_and_samples(self):
        th.manual_seed(0)
        model = FixedLogits([1.0, 2.0, 3.0])
        controller = StochasticController(model, num_actions=3)
        obs = th.zeros(5, 2)

        probs = controller.probabilities(obs)
        actions = controller.choose(obs)

        self.assertEqual(probs.shape, (5, 3))
        self.assertTrue(th.allclose(probs.sum(dim=-1), th.ones(5)))
        self.assertEqual(actions.shape, (5,))
        self.assertTrue(th.all((0 <= actions) & (actions < 3)))

    def test_warmup_controller_samples_actions_before_delegating(self):
        model = FixedLogits([1.0, 0.0, 0.0])
        greedy = GreedyController(model, num_actions=3)
        controller = WarmupController(greedy, FixedActionSpace(2), warmup_steps=2)
        obs = th.zeros(1, 2)

        self.assertIsInstance(controller, ContinuousActionController)
        self.assertIsInstance(controller, DiscreteActionController)
        self.assertTrue(th.equal(controller.choose(obs), th.tensor([2])))
        self.assertTrue(th.equal(controller.choose(obs), th.tensor([2])))
        self.assertTrue(th.equal(controller.choose(obs), th.tensor([0])))
        self.assertEqual(controller.steps, 2)

    def test_warmup_controller_returns_uniform_discrete_probabilities(self):
        model = FixedLogits([1.0, 0.0, 0.0])
        greedy = GreedyController(model, num_actions=3)
        controller = WarmupController(greedy, FixedActionSpace(2), warmup_steps=1)

        probs = controller.probabilities(th.zeros(1, 2))

        self.assertTrue(th.allclose(probs, th.full((1, 3), 1 / 3)))

    def test_warmup_controller_samples_continuous_actions(self):
        model = FixedGaussianOutput([0.0, 0.0, -3.0, -3.0])
        gaussian = GaussianController(model, action_dim=2, deterministic=True)
        action_space = FixedActionSpace(np.asarray([0.5, -0.25], dtype=np.float32))
        controller = WarmupController(gaussian, action_space, warmup_steps=1)
        obs = th.zeros(2, 2)

        warmup_actions = controller.choose(obs)
        policy_actions = controller.choose(obs)

        self.assertEqual(warmup_actions.shape, (2, 2))
        self.assertTrue(th.allclose(warmup_actions[0], th.tensor([0.5, -0.25])))
        self.assertTrue(th.allclose(policy_actions, th.zeros(2, 2)))
