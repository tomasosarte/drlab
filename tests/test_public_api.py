import unittest

import drlab


class PublicAPITest(unittest.TestCase):
    def test_root_exports_expected_public_symbols(self):
        expected = {
            "ActorCritic",
            "ActorCriticConfig",
            "ActorCriticExperiment",
            "ActorCriticExperimentConfig",
            "Controller",
            "DQN",
            "DQNConfig",
            "DQNExperiment",
            "DQNExperimentConfig",
            "EpsilonGreedyController",
            "GreedyController",
            "ReplayBuffer",
            "Runner",
            "StochasticController",
            "TransitionBatch",
        }

        self.assertEqual(drlab.__version__, "0.1.0")
        self.assertTrue(expected.issubset(set(drlab.__all__)))
        for name in expected:
            self.assertTrue(hasattr(drlab, name), name)
