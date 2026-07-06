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
            dones=th.tensor([[False], [True], [False]]),
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
            config=SACConfig(action_shape=(action_dim,)),
        )
        before = [param.detach().clone() for param in actor.parameters()]

        loss = learner.train(
            rewards=th.tensor([[1.0], [0.0], [1.0]]),
            dones=th.tensor([[False], [True], [False]]),
            states=th.randn(3, obs_dim),
            actions=th.randn(3, action_dim).tanh(),
            next_states=th.randn(3, obs_dim),
        )

        self.assertIsInstance(loss, float)
        self.assertTrue(th.isfinite(th.tensor(loss)))
        self.assertTrue(parameters_changed(actor, before))
        self.assertEqual(
            set(learner.last_losses),
            {"actor", "critic", "alpha", "total"},
        )

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
            dones=th.tensor([[False], [False], [True]]),
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
            dones=th.tensor([[False], [False], [True]]),
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
            dones=th.tensor([[False], [False], [True]]),
            states=th.randn(3, 4),
            actions=th.tensor([[0], [1], [0]], dtype=th.long),
            next_states=th.randn(3, 4),
            returns=th.tensor([[1.4], [0.5], [0.0]]),
        )

        self.assertIsInstance(loss, float)
        self.assertTrue(th.isfinite(th.tensor(loss)))
        self.assertTrue(parameters_changed(actor, before))
