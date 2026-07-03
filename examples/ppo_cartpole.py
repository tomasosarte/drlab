import gymnasium as gym
import torch as th

from drlab import (
    OnPolicyExperiment,
    OnPolicyExperimentConfig,
    PPOConfig,
    PPOLearner,
    StochasticController,
    ValueTargets,
)


def main(max_steps: int = 10_000, log_dir: str = "runs/examples/ppo_cartpole"):
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = th.nn.Sequential(
        th.nn.Linear(obs_dim, 64),
        th.nn.ReLU(),
        th.nn.Linear(64, num_actions + 1),
    )
    optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

    learner = PPOLearner(
        model,
        optimizer,
        PPOConfig(
            num_actions=num_actions,
            value_targets=ValueTargets.RETURNS,
            advantage_bootstrap=False,
            normalize_advantages=True,
            value_lambda=0.5,
            ppo_iterations=4,
        ),
    )
    controller = StochasticController(model, num_actions=num_actions)

    experiment = OnPolicyExperiment(
        env,
        controller,
        learner,
        OnPolicyExperimentConfig(
            max_steps=max_steps,
            run_steps=0,
            log_dir=log_dir,
            experiment_name="PPO CartPole",
        ),
    )
    experiment.run()
    env.close()


if __name__ == "__main__":
    main()
