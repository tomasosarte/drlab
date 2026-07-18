import unittest

import torch as th

from drlab.learners import (
    ActorCriticConfig,
    ActorCriticLearner,
    DQNConfig,
    DQNLearner,
    OffPolicyLearner,
    PPOConfig,
    PPOLearner,
    ReinforceConfig,
    ReinforceLearner,
    SACConfig,
    SACLearner,
)


def parameters_changed(model, before):
    return any(
        not th.allclose(param.detach(), old)
        for param, old in zip(model.parameters(), before)
    )


class LearnerSmokeTest(unittest.TestCase):
    def test_soft_target_update_matches_original_arithmetic_exactly(self):
        source1 = th.nn.Linear(4, 3)
        source2 = th.nn.Linear(3, 2)
        target1 = th.nn.Linear(4, 3)
        target2 = th.nn.Linear(3, 2)
        expected1 = th.nn.Linear(4, 3)
        expected2 = th.nn.Linear(3, 2)
        expected1.load_state_dict(target1.state_dict())
        expected2.load_state_dict(target2.state_dict())

        model = th.nn.Linear(1, 2)
        learner = DQNLearner(
            model,
            th.optim.SGD(model.parameters(), lr=0.1),
            DQNConfig(num_actions=2, soft_target_update_param=0.005),
        )

        with th.no_grad():
            for target, source in ((expected1, source1), (expected2, source2)):
                for target_param, source_param in zip(
                    target.parameters(), source.parameters()
                ):
                    target_param.data.mul_(0.995)
                    target_param.data.add_(0.005 * source_param.data)

        learner.update_targets([(target1, source1), (target2, source2)])

        for actual, expected in zip(
            [*target1.parameters(), *target2.parameters()],
            [*expected1.parameters(), *expected2.parameters()],
        ):
            self.assertTrue(th.equal(actual, expected))

    def test_dqn_train_step_returns_float_and_updates_parameters(self):
        th.manual_seed(0)
        model = th.nn.Sequential(
            th.nn.Linear(4, 8),
            th.nn.ReLU(),
            th.nn.Linear(8, 2),
        )
        optimizer = th.optim.SGD(model.parameters(), lr=0.05)
        learner = DQNLearner(
            model,
            optimizer,
            DQNConfig(
                num_actions=2,
                double_q=True,
                use_target_model=True,
                soft_target_update_param=0.5,
            ),
        )
        before = [param.detach().clone() for param in model.parameters()]

        loss = learner.train(
            rewards=th.tensor([[1.0], [0.0], [1.0]]),
            terminated=th.tensor([[False], [True], [False]]),
            states=th.randn(3, 4),
            actions=th.tensor([[0], [1], [0]], dtype=th.long),
            next_states=th.randn(3, 4),
        )

        self.assertIsInstance(loss, float)
        self.assertIsInstance(learner, OffPolicyLearner)
        self.assertTrue(th.isfinite(th.tensor(loss)))
        self.assertTrue(parameters_changed(model, before))

    def test_sac_train_step_returns_float_and_updates_parameters(self):
        th.manual_seed(0)
        obs_dim = 4
        action_dim = 2
        actor = th.nn.Sequential(
            th.nn.Linear(obs_dim, 8),
            th.nn.ReLU(),
            th.nn.Linear(8, 2 * action_dim),
        )
        critic1 = th.nn.Sequential(
            th.nn.Linear(obs_dim + action_dim, 8),
            th.nn.ReLU(),
            th.nn.Linear(8, 1),
        )
        critic2 = th.nn.Sequential(
            th.nn.Linear(obs_dim + action_dim, 8),
            th.nn.ReLU(),
            th.nn.Linear(8, 1),
        )
        learner = SACLearner(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optimizer=th.optim.SGD(actor.parameters(), lr=0.05),
            critic1_optimizer=th.optim.SGD(critic1.parameters(), lr=0.05),
            critic2_optimizer=th.optim.SGD(critic2.parameters(), lr=0.05),
            config=SACConfig(action_shape=(action_dim,), initial_alpha=0.2),
        )
        actor_before = [param.detach().clone() for param in actor.parameters()]
        critic1_before = [param.detach().clone() for param in critic1.parameters()]
        critic2_before = [param.detach().clone() for param in critic2.parameters()]

        self.assertAlmostEqual(learner.alpha.item(), 0.2)

        probe_states = th.randn(3, obs_dim)
        th.manual_seed(123)
        sampled_actions, sampled_log_probs = learner.sample_action_and_log_prob(
            probe_states
        )
        th.manual_seed(123)
        mean, std = learner.get_policy_dist(probe_states)
        dist = th.distributions.Normal(mean, std, validate_args=False)
        unsquashed_actions = dist.rsample()
        expected_actions = th.tanh(unsquashed_actions)
        expected_log_probs = dist.log_prob(unsquashed_actions)
        expected_log_probs -= th.log(1.0 - expected_actions.pow(2) + 1e-6)
        expected_log_probs = expected_log_probs.sum(dim=-1, keepdim=True)

        self.assertTrue(th.equal(sampled_actions, expected_actions))
        self.assertTrue(
            th.allclose(sampled_log_probs, expected_log_probs, atol=1e-6, rtol=1e-5)
        )

        loss = learner.train(
            rewards=th.tensor([[1.0], [0.0], [1.0]]),
            terminated=th.tensor([[False], [True], [False]]),
            states=th.randn(3, obs_dim),
            actions=th.randn(3, action_dim).tanh(),
            next_states=th.randn(3, obs_dim),
        )

        self.assertIsInstance(loss, float)
        self.assertTrue(th.isfinite(th.tensor(loss)))
        self.assertTrue(parameters_changed(actor, actor_before))
        self.assertTrue(parameters_changed(critic1, critic1_before))
        self.assertTrue(parameters_changed(critic2, critic2_before))
        self.assertEqual(
            set(learner.last_losses),
            {"actor", "critic", "alpha", "regularization", "total"},
        )

    def test_sac_initial_alpha_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "initial_alpha must be > 0"):
            SACConfig(initial_alpha=0.0)

    def test_sac_actor_regularizer_contributes_to_loss(self):
        th.manual_seed(0)
        obs_dim = 4
        action_dim = 2
        actor = th.nn.Sequential(
            th.nn.Linear(obs_dim, 8),
            th.nn.ReLU(),
            th.nn.Linear(8, 2 * action_dim),
        )
        critic1 = th.nn.Sequential(
            th.nn.Linear(obs_dim + action_dim, 8),
            th.nn.ReLU(),
            th.nn.Linear(8, 1),
        )
        critic2 = th.nn.Sequential(
            th.nn.Linear(obs_dim + action_dim, 8),
            th.nn.ReLU(),
            th.nn.Linear(8, 1),
        )
        calls = []

        def actor_l2(model, rewards, terminated, states, actions, next_states):
            calls.append(model)
            return sum(param.square().sum() for param in model.parameters())

        learner = SACLearner(
            actor=actor,
            critic1=critic1,
            critic2=critic2,
            actor_optimizer=th.optim.SGD(actor.parameters(), lr=0.05),
            critic1_optimizer=th.optim.SGD(critic1.parameters(), lr=0.05),
            critic2_optimizer=th.optim.SGD(critic2.parameters(), lr=0.05),
            config=SACConfig(
                action_shape=(action_dim,),
                regularizers=[actor_l2],
                reg_lams=[0.01],
            ),
        )

        learner.train(
            rewards=th.tensor([[1.0], [0.0], [1.0]]),
            terminated=th.tensor([[False], [True], [False]]),
            states=th.randn(3, obs_dim),
            actions=th.randn(3, action_dim).tanh(),
            next_states=th.randn(3, obs_dim),
        )

        self.assertEqual(calls, [actor])
        self.assertGreater(learner.last_losses["regularization"], 0.0)

    def test_reinforce_train_step_returns_float_and_updates_parameters(self):
        th.manual_seed(0)
        actor = th.nn.Sequential(
            th.nn.Linear(4, 8),
            th.nn.Tanh(),
            th.nn.Linear(8, 2),
        )
        optimizer = th.optim.SGD(actor.parameters(), lr=0.05)
        learner = ReinforceLearner(
            actor,
            optimizer,
            ReinforceConfig(num_actions=2, normalize_returns=True),
        )
        before = [param.detach().clone() for param in actor.parameters()]

        loss = learner.train(
            rewards=th.tensor([[1.0], [0.5], [0.0]]),
            terminated=th.tensor([[False], [False], [True]]),
            states=th.randn(3, 4),
            actions=th.tensor([[0], [1], [0]], dtype=th.long),
            next_states=th.randn(3, 4),
            returns=th.tensor([[1.4], [0.5], [0.0]]),
        )

        self.assertIsInstance(loss, float)
        self.assertTrue(th.isfinite(th.tensor(loss)))
        self.assertTrue(parameters_changed(actor, before))

    def test_actor_critic_train_step_returns_float_and_updates_parameters(self):
        th.manual_seed(0)
        actor = th.nn.Sequential(
            th.nn.Linear(4, 8),
            th.nn.Tanh(),
            th.nn.Linear(8, 3),
        )
        optimizer = th.optim.SGD(actor.parameters(), lr=0.05)
        learner = ActorCriticLearner(
            actor,
            optimizer,
            ActorCriticConfig(num_actions=2, value_lambda=0.2),
        )
        before = [param.detach().clone() for param in actor.parameters()]

        loss = learner.train(
            rewards=th.tensor([[1.0], [0.5], [0.0]]),
            terminated=th.tensor([[False], [False], [True]]),
            states=th.randn(3, 4),
            actions=th.tensor([[0], [1], [0]], dtype=th.long),
            next_states=th.randn(3, 4),
            returns=th.tensor([[1.4], [0.5], [0.0]]),
        )

        self.assertIsInstance(loss, float)
        self.assertTrue(th.isfinite(th.tensor(loss)))
        self.assertTrue(parameters_changed(actor, before))

    def test_ppo_train_step_returns_float_and_updates_parameters(self):
        th.manual_seed(0)
        actor = th.nn.Sequential(
            th.nn.Linear(4, 8),
            th.nn.Tanh(),
            th.nn.Linear(8, 3),
        )
        optimizer = th.optim.SGD(actor.parameters(), lr=0.05)
        learner = PPOLearner(
            actor,
            optimizer,
            PPOConfig(num_actions=2, value_lambda=0.2, ppo_iterations=2),
        )
        before = [param.detach().clone() for param in actor.parameters()]

        loss = learner.train(
            rewards=th.tensor([[1.0], [0.5], [0.0]]),
            terminated=th.tensor([[False], [False], [True]]),
            states=th.randn(3, 4),
            actions=th.tensor([[0], [1], [0]], dtype=th.long),
            next_states=th.randn(3, 4),
            returns=th.tensor([[1.4], [0.5], [0.0]]),
        )

        self.assertIsInstance(loss, float)
        self.assertTrue(th.isfinite(th.tensor(loss)))
        self.assertTrue(parameters_changed(actor, before))
