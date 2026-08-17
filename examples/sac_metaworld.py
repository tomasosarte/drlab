"""Train drlab SAC on one MetaWorld task with a Continual-World-style setup.

The default configuration mirrors the single-task experiment used by
Continual World: four 256-unit hidden layers, LayerNorm, LeakyReLU, batches of
128 transitions, and 50 gradient updates after every 50 environment steps.

MetaWorld is an optional dependency. Install it before running this example:

    python -m pip install metaworld
    python examples/sac_metaworld.py --task reach
"""

import argparse
import random
import time
from collections.abc import Iterable
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

from drlab import (
    GaussianController,
    OffPolicyExperiment,
    OffPolicyExperimentConfig,
    SACConfig,
    SACLearner,
)


ACTIVATIONS: dict[str, type[th.nn.Module]] = {
    "lrelu": th.nn.LeakyReLU,
    "relu": th.nn.ReLU,
    "tanh": th.nn.Tanh,
}


class HalfMSELoss(th.nn.Module):
    """MSE scaled like the original Continual World SAC critic loss."""

    def forward(self, prediction: th.Tensor, target: th.Tensor) -> th.Tensor:
        return 0.5 * th.nn.functional.mse_loss(prediction, target)


def mlp(
    input_dim: int,
    hidden_sizes: Iterable[int],
    activation: type[th.nn.Module],
    layer_norm: bool,
) -> th.nn.Sequential:
    layers: list[th.nn.Module] = []
    previous_dim = input_dim

    for hidden_dim in hidden_sizes:
        layers.append(th.nn.Linear(previous_dim, hidden_dim))
        if layer_norm:
            layers.append(th.nn.LayerNorm(hidden_dim))
        layers.append(activation())
        previous_dim = hidden_dim

    if not layers:
        raise ValueError("hidden_sizes must contain at least one layer.")

    return th.nn.Sequential(*layers)


class MlpActor(th.nn.Module):
    """SAC actor that returns concatenated Gaussian mean and log standard deviation."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int],
        activation: type[th.nn.Module],
        layer_norm: bool,
    ):
        super().__init__()
        hidden_sizes = tuple(hidden_sizes)
        self.core = mlp(
            observation_dim,
            hidden_sizes,
            activation,
            layer_norm,
        )
        self.mean = th.nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = th.nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        features = self.core(observations)
        return th.cat([self.mean(features), self.log_std(features)], dim=-1)


class MlpCritic(th.nn.Module):
    """SAC critic that consumes one concatenated observation-action tensor."""

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_sizes: Iterable[int],
        activation: type[th.nn.Module],
        layer_norm: bool,
    ):
        super().__init__()
        hidden_sizes = tuple(hidden_sizes)
        self.core = mlp(
            observation_dim + action_dim,
            hidden_sizes,
            activation,
            layer_norm,
        )
        self.value = th.nn.Linear(hidden_sizes[-1], 1)

    def forward(self, state_actions: th.Tensor) -> th.Tensor:
        return self.value(self.core(state_actions))


class OneHotSingleTaskObservation(gym.ObservationWrapper):
    """Append the constant one-element task identifier used by Continual World."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        observation_space = env.observation_space
        if not isinstance(observation_space, spaces.Box):
            raise TypeError("This example requires a Box observation space.")

        self.observation_space = spaces.Box(
            low=np.concatenate(
                [
                    observation_space.low.astype(np.float32),
                    np.zeros(1, dtype=np.float32),
                ]
            ),
            high=np.concatenate(
                [
                    observation_space.high.astype(np.float32),
                    np.ones(1, dtype=np.float32),
                ]
            ),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate(
            [observation.astype(np.float32, copy=False), np.ones(1, dtype=np.float32)]
        )


class SuccessCounter(gym.Wrapper):
    """Record whether each completed episode reached a successful state."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.successes: list[bool] = []
        self._episode_success = False

    def reset(self, **kwargs: Any):
        self._episode_success = False
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._episode_success |= bool(info.get("success", False))
        if terminated or truncated:
            self.successes.append(self._episode_success)
        return observation, reward, terminated, truncated, info

    def pop_successes(self) -> list[bool]:
        successes = self.successes
        self.successes = []
        return successes


def normalize_task_name(task: str) -> str:
    name = task.lower().replace("_", "-")
    for suffix in ("-v1", "-v2", "-v3"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return f"{name}-v3"


def make_env(task: str, seed: int, max_episode_len: int) -> SuccessCounter:
    try:
        import metaworld  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MetaWorld is not installed. Install it with "
            "`python -m pip install metaworld`."
        ) from exc

    task_name = normalize_task_name(task)
    env = gym.make(
        "Meta-World/MT1",
        env_name=task_name,
        seed=seed,
        disable_env_checker=True,
    )
    env = OneHotSingleTaskObservation(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_len)
    env = SuccessCounter(env)
    env.action_space.seed(seed)
    return env


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if th.cuda.is_available() else "cpu"
    if device == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def use_fused_adam(optimizer: str, device: str) -> bool:
    if optimizer == "auto":
        return device == "cuda"
    if optimizer == "fused-adam":
        if device != "cuda":
            raise ValueError("fused-adam requires a CUDA device.")
        return True
    return False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train drlab SAC on one MetaWorld task."
    )
    parser.add_argument("--task", default="reach")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[256, 256, 256, 256],
    )
    parser.add_argument(
        "--activation",
        choices=sorted(ACTIVATIONS),
        default="lrelu",
    )
    parser.add_argument(
        "--layer-norm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-episode-len", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--optimizer",
        choices=("auto", "adam", "fused-adam"),
        default="auto",
        help="'auto' uses fused Adam on CUDA and regular Adam on CPU.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-entropy", type=float, default=None)
    parser.add_argument("--alpha-lr", type=float, default=1e-3)
    parser.add_argument("--clipnorm", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-buffer-size", type=int, default=1_000_000)
    parser.add_argument(
        "--update-every",
        type=int,
        default=50,
        help="Collect and then train for this many steps.",
    )
    parser.add_argument("--warmup-steps", type=int, default=10_000)
    parser.add_argument("--learning-starts", type=int, default=1_000)
    parser.add_argument("--log-dir", default="runs/examples/sac_metaworld")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    env = make_env(args.task, args.seed, args.max_episode_len)

    observation_dim = int(np.prod(env.observation_space.shape))
    action_shape = env.action_space.shape
    action_dim = int(np.prod(action_shape))
    hidden_sizes = tuple(args.hidden_sizes)
    activation = ACTIVATIONS[args.activation]
    fused_adam = use_fused_adam(args.optimizer, device)

    actor = MlpActor(
        observation_dim,
        action_dim,
        hidden_sizes,
        activation,
        args.layer_norm,
    )
    critic1 = MlpCritic(
        observation_dim,
        action_dim,
        hidden_sizes,
        activation,
        args.layer_norm,
    )
    critic2 = MlpCritic(
        observation_dim,
        action_dim,
        hidden_sizes,
        activation,
        args.layer_norm,
    )

    learner = SACLearner(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optimizer=th.optim.Adam(
            actor.parameters(),
            lr=args.lr,
            fused=fused_adam,
        ),
        critic_optimizer=th.optim.Adam(
            [*critic1.parameters(), *critic2.parameters()],
            lr=args.lr,
            fused=fused_adam,
        ),
        config=SACConfig(
            device=device,
            gamma=args.gamma,
            action_shape=action_shape,
            soft_target_update_param=args.tau,
            target_entropy=args.target_entropy,
            alpha_lr=args.alpha_lr,
            initial_alpha=np.e,
            clipnorm=args.clipnorm,
            criterion=HalfMSELoss(),
        ),
    )
    experiment = OffPolicyExperiment(
        env=env,
        controller=GaussianController(actor, action_dim=action_dim),
        learner=learner,
        config=OffPolicyExperimentConfig(
            max_steps=args.steps,
            gamma=args.gamma,
            run_steps=args.update_every,
            log_dir=args.log_dir,
            experiment_name=f"SAC-{normalize_task_name(args.task)}",
            replay_buffer_size=args.replay_buffer_size,
            batch_size=args.batch_size,
            use_last_episode=False,
            grad_repeats=args.update_every,
            warmup_steps=args.warmup_steps,
            learning_starts=args.learning_starts,
        ),
    )

    print(
        f"Running SAC on {normalize_task_name(args.task)}: "
        f"obs_dim={observation_dim}, action_dim={action_dim}, steps={args.steps}, "
        f"update_every={args.update_every}, optimizer="
        f"{'fused-adam' if fused_adam else 'adam'}, device={device}"
    )

    started = time.perf_counter()
    try:
        experiment.run()
        elapsed = time.perf_counter() - started
        print(
            f"Finished in {elapsed:.2f}s "
            f"({args.steps / elapsed:.1f} environment steps/s)."
        )
        if learner.last_losses:
            rounded_losses = {
                name: round(value, 4) for name, value in learner.last_losses.items()
            }
            print("Final losses:", rounded_losses)

        successes = env.pop_successes()
        if successes:
            print(f"Episode success rate: {float(np.mean(successes)):.3f}")
        else:
            print("No complete episodes were recorded during this short run.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
