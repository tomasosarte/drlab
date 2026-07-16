import unittest

import drlab


class PublicAPITest(unittest.TestCase):
    def test_root_exports_expected_public_symbols(self):
        expected = {
            "ActorCriticConfig",
            "ActorCriticLearner",
            "OnPolicyExperiment",
            "OnPolicyExperimentConfig",
            "ContinuousActionController",
            "Controller",
            "DiscreteActionController",
            "DQNConfig",
            "DQNLearner",
            "OnPolicyConfig",
            "OnPolicyLearner",
            "OffPolicyConfig",
            "OffPolicyExperiment",
            "OffPolicyExperimentConfig",
            "OffPolicyLearner",
            "PPOConfig",
            "PPOLearner",
            "ReinforceConfig",
            "ReinforceLearner",
            "TargetUpdate",
            "EpsilonGreedyController",
            "GaussianController",
            "GreedyController",
            "ReplayBuffer",
            "Runner",
            "StochasticController",
            "TransitionBatch",
            "ValueTargets",
            "WarmupController",
        }

        self.assertEqual(drlab.__version__, "0.2.3")
        self.assertTrue(expected.issubset(set(drlab.__all__)))
        for name in expected:
            self.assertTrue(hasattr(drlab, name), name)
