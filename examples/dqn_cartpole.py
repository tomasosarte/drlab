import gymnasium as gym
import torch as th

from drlab import (
    DQNConfig,
    DQNLearner,
    EpsilonGreedyController,
    GreedyController,
    OffPolicyExperiment,
    OffPolicyExperimentConfig,
)


def main(max_steps: int = 10_000, log_dir: str = "runs/examples/dqn_cartpole"):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = th.nn.Sequential(
        th.nn.Linear(obs_dim, 64),
        th.nn.ReLU(),
        th.nn.Linear(64, num_actions),
    )
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

    learner = DQNLearner(
        model,
        optimizer,
        DQNConfig(num_actions=num_actions),
    )
    controller = EpsilonGreedyController(
        GreedyController(model, num_actions=num_actions),
        num_actions=num_actions,
        max_eps=1.0,
        min_eps=0.05,
        anneal_steps=max_steps,
    )

    experiment = OffPolicyExperiment(
        env,
        controller,
        learner,
        OffPolicyExperimentConfig(
            max_steps=max_steps,
            run_steps=1,
            batch_size=64,
            log_dir=log_dir,
            experiment_name="DQN CartPole",
        ),
    )
    experiment.run()
    env.close()


if __name__ == "__main__":
    main()
