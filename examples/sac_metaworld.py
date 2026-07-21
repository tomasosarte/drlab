import argparse

import gymnasium as gym
import numpy as np
import torch as th

from drlab import (
    GaussianController,
    OffPolicyExperiment,
    OffPolicyExperimentConfig,
    SACConfig,
    SACLearner,
)


def mlp(input_dim: int, output_dim: int, hidden_dim: int) -> th.nn.Sequential:
    return th.nn.Sequential(
        th.nn.Flatten(),
        th.nn.Linear(input_dim, hidden_dim),
        th.nn.ReLU(),
        th.nn.Linear(hidden_dim, hidden_dim),
        th.nn.ReLU(),
        th.nn.Linear(hidden_dim, output_dim),
    )


class QNetwork(th.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = mlp(obs_dim + action_dim, 1, hidden_dim)

    def forward(self, state_actions: th.Tensor) -> th.Tensor:
        return self.net(state_actions)


def make_env(task_name: str, seed: int) -> gym.Env:
    try:
        import metaworld  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MetaWorld is not installed. Install it with "
            "`python -m pip install metaworld`."
        ) from exc

    env = gym.make("Meta-World/MT1", env_name=task_name, seed=seed)
    env.action_space.seed(seed)
    return env


def main(
    task_name: str = "reach-v3",
    max_steps: int = 50_000,
    seed: int = 0,
    batch_size: int = 256,
    replay_size: int = 100_000,
    grad_repeats: int = 1,
    hidden_dim: int = 256,
    device: str | None = None,
    log_dir: str | None = None,
):
    np.random.seed(seed)
    th.manual_seed(seed)
    device = device or ("cuda" if th.cuda.is_available() else "cpu")

    env = make_env(task_name, seed)
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("This SAC example expects a Box observation space.")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("This SAC example expects a Box action space.")
    if not (
        np.allclose(env.action_space.low, -1.0)
        and np.allclose(env.action_space.high, 1.0)
    ):
        raise ValueError("This SAC example expects actions bounded in [-1, 1].")

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    actor = mlp(obs_dim, 2 * action_dim, hidden_dim)
    critic1 = QNetwork(obs_dim, action_dim, hidden_dim)
    critic2 = QNetwork(obs_dim, action_dim, hidden_dim)

    learner = SACLearner(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optimizer=th.optim.Adam(actor.parameters(), lr=3e-4),
        critic_optimizer=th.optim.Adam(
            [*critic1.parameters(), *critic2.parameters()],
            lr=3e-4,
        ),
        config=SACConfig(
            device=device,
            action_shape=env.action_space.shape,
            gamma=0.99,
            soft_target_update_param=0.005,
        ),
    )
    controller = GaussianController(actor, action_dim=action_dim)

    safe_task_name = task_name.replace("/", "_")
    experiment = OffPolicyExperiment(
        env,
        controller,
        learner,
        OffPolicyExperimentConfig(
            max_steps=max_steps,
            gamma=0.99,
            run_steps=1,
            batch_size=batch_size,
            replay_buffer_size=replay_size,
            warmup_steps=1_000,
            use_last_episode=False,
            grad_repeats=grad_repeats,
            log_dir=log_dir or f"runs/examples/sac_metaworld_{safe_task_name}",
            experiment_name=f"SAC MetaWorld {task_name}",
        ),
    )

    try:
        experiment.run()
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on a MetaWorld MT1 task.")
    parser.add_argument("--task", default="reach-v3", help="MetaWorld task name.")
    parser.add_argument("--steps", type=int, default=50_000, help="Environment steps.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=100_000)
    parser.add_argument("--grad-repeats", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        task_name=args.task,
        max_steps=args.steps,
        seed=args.seed,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        grad_repeats=args.grad_repeats,
        hidden_dim=args.hidden_dim,
        device=args.device,
        log_dir=args.log_dir,
    )
