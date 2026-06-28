import unittest

import drlab


class PublicAPITest(unittest.TestCase):
    def test_root_exports_expected_public_symbols(self):
        expected = {
            "ActorCritic",
            "ActorCriticConfig",
            "OnPolicyExperiment",
            "OnPolicyExperimentConfig",
            "Controller",
            "DQN",
            "DQNConfig",
            "OffPolicyExperiment",
            "OffPolicyExperimentConfig",
            "EpsilonGreedyController",
            "GreedyController",
            "ReplayBuffer",
            "Runner",
            "StochasticController",
            "TransitionBatch",
        }

        self.assertEqual(drlab.__version__, "0.1.1")
        self.assertTrue(expected.issubset(set(drlab.__all__)))
        for name in expected:
            self.assertTrue(hasattr(drlab, name), name)
