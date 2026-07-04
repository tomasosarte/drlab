from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch as th

from drlab import (
    DQNConfig,
    DQNLearner,
    EpsilonGreedyController,
    GreedyController,
    OffPolicyExperiment,
    OffPolicyExperimentConfig,
)


ENV_ID = "CartPole-v1"


@dataclass
class EvalResult:
    mean: float
    std: float
    returns: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare drlab DQN with Stable-Baselines3 DQN on CartPole-v1.",
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-dir",
        default="runs/examples/compare_drlab_stable_baselines3",
    )
    return parser.parse_args()


def require_stable_baselines3():
    if importlib.util.find_spec("stable_baselines3") is None:
        raise SystemExit(
            "This example needs Stable-Baselines3. Install it with:\n"
            "  python -m pip install stable-baselines3"
        )


def set_seed(seed: int):
    np.random.seed(seed)
    th.manual_seed(seed)


def make_env(seed: int | None = None):
    env = gym.make(ENV_ID)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env


def make_q_network(obs_dim: int, num_actions: int) -> th.nn.Module:
    return th.nn.Sequential(
        th.nn.Linear(obs_dim, 64),
        th.nn.ReLU(),
        th.nn.Linear(64, 64),
        th.nn.ReLU(),
        th.nn.Linear(64, num_actions),
    )


def train_drlab(total_steps: int, seed: int, log_dir: str) -> th.nn.Module:
    set_seed(seed)
    env = make_env(seed)
    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = make_q_network(obs_dim, num_actions)
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
        anneal_steps=max(total_steps, 2),
    )

    experiment = OffPolicyExperiment(
        env,
        controller,
        learner,
        OffPolicyExperimentConfig(
            max_steps=total_steps,
            run_steps=1,
            batch_size=64,
            replay_buffer_size=10_000,
            use_last_episode=False,
            log_dir=f"{log_dir}/drlab",
            experiment_name="drlab DQN CartPole",
        ),
    )
    experiment.run()
    env.close()
    return model


def train_stable_baselines3(total_steps: int, seed: int, log_dir: str):
    from stable_baselines3 import DQN

    env = make_env(seed)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10_000,
        learning_starts=64,
        batch_size=64,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=250,
        exploration_fraction=0.1,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [64, 64]},
        tensorboard_log=f"{log_dir}/stable_baselines3",
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=total_steps, tb_log_name="dqn_cartpole")
    env.close()
    return model


def evaluate_drlab(model: th.nn.Module, episodes: int, seed: int) -> EvalResult:
    model.eval()
    controller = GreedyController(model, num_actions=2)
    returns = []

    env = make_env()
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0
        while not done:
            obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
            with th.inference_mode():
                action = int(controller.choose(obs_t).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += float(reward)
        returns.append(episode_return)
    env.close()

    return EvalResult(float(np.mean(returns)), float(np.std(returns)), returns)


def evaluate_stable_baselines3(model, episodes: int, seed: int) -> EvalResult:
    returns = []

    env = make_env()
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action = int(np.asarray(action).item())
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += float(reward)
        returns.append(episode_return)
    env.close()

    return EvalResult(float(np.mean(returns)), float(np.std(returns)), returns)


def print_result(name: str, result: EvalResult):
    print(f"{name:22s} mean={result.mean:7.2f}  std={result.std:6.2f}")
    print(f"{'':22s} returns={result.returns}")


def main():
    args = parse_args()
    require_stable_baselines3()

    print(f"Training each agent for {args.steps} steps on {ENV_ID}.")

    print("\nTraining drlab...")
    drlab_model = train_drlab(args.steps, args.seed, args.log_dir)
    drlab_result = evaluate_drlab(
        drlab_model,
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
    )

    print("\nTraining Stable-Baselines3...")
    sb3_model = train_stable_baselines3(args.steps, args.seed, args.log_dir)
    sb3_result = evaluate_stable_baselines3(
        sb3_model,
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
    )

    print(f"\nEvaluation over {args.eval_episodes} deterministic episodes:")
    print_result("drlab DQN", drlab_result)
    print_result("Stable-Baselines3 DQN", sb3_result)
    print(f"\nTensorBoard logs: {args.log_dir}")


if __name__ == "__main__":
    main()
