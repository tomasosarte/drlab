from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import random
import time
from contextlib import contextmanager, redirect_stderr
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.wrappers import RescaleAction

from drlab import (
    ActorCriticConfig,
    ActorCriticLearner,
    DQNConfig,
    DQNLearner,
    EpsilonGreedyController,
    GaussianController,
    GreedyController,
    OffPolicyExperiment,
    OffPolicyExperimentConfig,
    OnPolicyExperiment,
    OnPolicyExperimentConfig,
    PPOConfig,
    PPOLearner,
    ReinforceConfig,
    ReinforceLearner,
    SACConfig,
    SACLearner,
    StochasticController,
    TargetUpdate,
    ValueTargets,
)


LEARNER_ORDER = ("dqn", "reinforce", "actor_critic", "ppo", "sac")
LEARNER_LABELS = {
    "dqn": "DQN",
    "reinforce": "REINFORCE",
    "actor_critic": "Actor-Critic",
    "ppo": "PPO",
    "sac": "SAC",
}

DRLAB = "drlab"
SB3 = "Stable-Baselines3"

HIDDEN_SIZES = (64, 64)
PENDULUM_MIN_STEP_RETURN = -(math.pi**2 + 0.1 * 8.0**2 + 0.001 * 2.0**2)
PENDULUM_MIN_RETURN = 200 * PENDULUM_MIN_STEP_RETURN


@dataclass(frozen=True)
class EnvSpec:
    env_id: str
    min_return: float
    max_return: float
    normalize_actions: bool = False

    def accuracy_pct(self, mean_return: float) -> float:
        span = self.max_return - self.min_return
        normalized = 100.0 * (mean_return - self.min_return) / span
        return float(np.clip(normalized, 0.0, 100.0))


CARTPOLE = EnvSpec("CartPole-v1", min_return=0.0, max_return=500.0)
PENDULUM = EnvSpec(
    "Pendulum-v1",
    min_return=PENDULUM_MIN_RETURN,
    max_return=0.0,
    normalize_actions=True,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    steps: int
    eval_episodes: int
    log_dir: Path
    device: str
    show_progress: bool
    gamma: float = 0.99
    learning_rate: float = 1e-3
    hidden_sizes: tuple[int, ...] = HIDDEN_SIZES
    dqn_batch_size: int = 64
    dqn_replay_size: int = 10_000
    dqn_target_update_interval: int = 250
    a2c_rollout_steps: int = 5
    ppo_rollout_steps: int = 128
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    sac_batch_size: int = 256
    sac_replay_size: int = 50_000
    sac_tau: float = 0.005


@dataclass(frozen=True)
class EvalResult:
    mean: float
    std: float
    returns: list[float]


@dataclass(frozen=True)
class RunResult:
    learner_key: str
    learner_label: str
    framework: str
    env: EnvSpec
    seed: int
    train_steps: int
    train_seconds: float
    eval: EvalResult

    @property
    def steps_per_second(self) -> float:
        return self.train_steps / max(self.train_seconds, 1e-12)


@dataclass
class SummaryRow:
    learner_key: str
    learner_label: str
    framework: str
    env_id: str
    seeds: int
    eval_mean: float
    eval_std: float
    accuracy_pct: float
    train_seconds: float
    steps_per_second: float
    speedup_vs_sb3: float | None = None


class StepCounter(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.step_count = 0

    def step(self, action: Any):
        self.step_count += 1
        return self.env.step(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare every drlab learner against matching Stable-Baselines3 "
            "baselines on simple environments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Run several seeds and aggregate results. Overrides --seed.",
    )
    parser.add_argument(
        "--learners",
        nargs="+",
        choices=LEARNER_ORDER,
        default=list(LEARNER_ORDER),
        help="Subset of drlab learners to benchmark.",
    )
    parser.add_argument(
        "--log-dir",
        default="runs/examples/compare_drlab_stable_baselines3",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device used by drlab and SB3.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Number of CPU threads for PyTorch. Use 0 to keep PyTorch defaults.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show drlab tqdm progress bars. Hidden by default for cleaner results.",
    )
    parser.add_argument(
        "--skip-sb3",
        action="store_true",
        help="Only run drlab learners and leave speedup columns blank.",
    )
    parser.add_argument(
        "--results-csv",
        default=None,
        help="Path for aggregate CSV results. Defaults to <log-dir>/results.csv.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write the aggregate CSV file.",
    )
    return parser.parse_args()


def require_stable_baselines3() -> None:
    if importlib.util.find_spec("stable_baselines3") is None:
        raise SystemExit(
            "This example needs Stable-Baselines3. Install it with:\n"
            "  python -m pip install stable-baselines3\n"
            "or pass --skip-sb3 to run only drlab learners."
        )


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0:
        raise SystemExit("--steps must be positive.")
    if args.eval_episodes <= 0:
        raise SystemExit("--eval-episodes must be positive.")
    if args.torch_threads < 0:
        raise SystemExit("--torch-threads must be >= 0.")


def selected_seeds(args: argparse.Namespace) -> list[int]:
    return args.seeds if args.seeds is not None else [args.seed]


def unique_learners(learners: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(learners))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.lower()).strip("_")


def make_env(
    spec: EnvSpec,
    seed: int | None = None,
    count_steps: bool = False,
) -> gym.Env:
    env = gym.make(spec.env_id)
    if spec.normalize_actions:
        min_action = np.full(env.action_space.shape, -1.0, dtype=np.float32)
        max_action = np.full(env.action_space.shape, 1.0, dtype=np.float32)
        env = RescaleAction(env, min_action=min_action, max_action=max_action)
    if count_steps:
        env = StepCounter(env)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def flat_box_dim(space: gym.Space) -> int:
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(f"Expected a Box space, got {type(space).__name__}.")
    return int(np.prod(space.shape))


def discrete_action_count(space: gym.Space) -> int:
    if not isinstance(space, gym.spaces.Discrete):
        raise TypeError(f"Expected a Discrete action space, got {type(space).__name__}.")
    return int(space.n)


def make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_sizes: Sequence[int],
) -> th.nn.Sequential:
    layers: list[th.nn.Module] = [th.nn.Flatten()]
    last_dim = input_dim
    for hidden_size in hidden_sizes:
        layers.append(th.nn.Linear(last_dim, hidden_size))
        layers.append(th.nn.ReLU())
        last_dim = hidden_size
    layers.append(th.nn.Linear(last_dim, output_dim))
    return th.nn.Sequential(*layers)


class QNetwork(th.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Sequence[int]):
        super().__init__()
        self.net = make_mlp(obs_dim + action_dim, 1, hidden_sizes)

    def forward(self, state_actions: th.Tensor) -> th.Tensor:
        return self.net(state_actions)


@contextmanager
def clean_training_output(show_progress: bool):
    if show_progress:
        yield
        return

    with open(os.devnull, "w") as sink, redirect_stderr(sink):
        yield


def timed_train(train_fn: Callable[[], None], show_progress: bool) -> float:
    start = time.perf_counter()
    with clean_training_output(show_progress):
        train_fn()
    return time.perf_counter() - start


def run_log_dir(config: BenchmarkConfig, learner_key: str, framework: str, seed: int) -> Path:
    return config.log_dir / learner_key / safe_name(framework) / f"seed_{seed}"


def eval_seed(seed: int) -> int:
    return seed + 100_000


def evaluate_discrete_model(
    model: th.nn.Module,
    env_spec: EnvSpec,
    num_actions: int,
    episodes: int,
    seed: int,
    device: str,
) -> EvalResult:
    model.eval()
    returns: list[float] = []
    env = make_env(env_spec)

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0

            while not done:
                obs_t = th.as_tensor(obs, dtype=th.float32, device=device).unsqueeze(0)
                with th.inference_mode():
                    logits = model(obs_t)[:, :num_actions]
                    action = int(th.argmax(logits, dim=-1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += float(reward)

            returns.append(episode_return)
    finally:
        env.close()

    return EvalResult(float(np.mean(returns)), float(np.std(returns)), returns)


def evaluate_sac_actor(
    actor: th.nn.Module,
    env_spec: EnvSpec,
    action_dim: int,
    episodes: int,
    seed: int,
    device: str,
) -> EvalResult:
    actor.eval()
    returns: list[float] = []
    env = make_env(env_spec)

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0

            while not done:
                obs_t = th.as_tensor(obs, dtype=th.float32, device=device).unsqueeze(0)
                with th.inference_mode():
                    output = actor(obs_t)
                    action = th.tanh(output[:, :action_dim])
                obs, reward, terminated, truncated, _ = env.step(
                    action.squeeze(0).cpu().numpy().astype(np.float32)
                )
                done = terminated or truncated
                episode_return += float(reward)

            returns.append(episode_return)
    finally:
        env.close()

    return EvalResult(float(np.mean(returns)), float(np.std(returns)), returns)


def evaluate_sb3_model(
    model: Any,
    env_spec: EnvSpec,
    episodes: int,
    seed: int,
) -> EvalResult:
    returns: list[float] = []
    env = make_env(env_spec)

    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_return += float(reward)

            returns.append(episode_return)
    finally:
        env.close()

    return EvalResult(float(np.mean(returns)), float(np.std(returns)), returns)


def close_sb3_env(model: Any) -> None:
    env = model.get_env()
    if env is not None:
        env.close()


def train_drlab_dqn(config: BenchmarkConfig, seed: int) -> RunResult:
    set_seed(seed)
    env = make_env(CARTPOLE, seed=seed, count_steps=True)
    obs_dim = flat_box_dim(env.observation_space)
    num_actions = discrete_action_count(env.action_space)

    model = make_mlp(obs_dim, num_actions, config.hidden_sizes)
    optimizer = th.optim.Adam(model.parameters(), lr=config.learning_rate)
    learner = DQNLearner(
        model,
        optimizer,
        DQNConfig(
            device=config.device,
            gamma=config.gamma,
            num_actions=num_actions,
            target_update=TargetUpdate.HARD,
            target_update_interval=config.dqn_target_update_interval,
        ),
    )
    controller = EpsilonGreedyController(
        GreedyController(model, num_actions=num_actions),
        num_actions=num_actions,
        max_eps=1.0,
        min_eps=0.05,
        anneal_steps=max(config.steps, 2),
    )
    experiment = OffPolicyExperiment(
        env,
        controller,
        learner,
        OffPolicyExperimentConfig(
            max_steps=config.steps,
            gamma=config.gamma,
            run_steps=1,
            batch_size=config.dqn_batch_size,
            replay_buffer_size=config.dqn_replay_size,
            use_last_episode=False,
            log_dir=str(run_log_dir(config, "dqn", DRLAB, seed)),
            experiment_name="drlab DQN CartPole",
        ),
    )

    try:
        seconds = timed_train(experiment.run, config.show_progress)
        train_steps = env.step_count
    finally:
        env.close()

    result = evaluate_discrete_model(
        model,
        CARTPOLE,
        num_actions,
        config.eval_episodes,
        eval_seed(seed),
        config.device,
    )
    return RunResult("dqn", LEARNER_LABELS["dqn"], DRLAB, CARTPOLE, seed, train_steps, seconds, result)


def train_drlab_reinforce(config: BenchmarkConfig, seed: int) -> RunResult:
    set_seed(seed)
    env = make_env(CARTPOLE, seed=seed, count_steps=True)
    obs_dim = flat_box_dim(env.observation_space)
    num_actions = discrete_action_count(env.action_space)

    model = make_mlp(obs_dim, num_actions, config.hidden_sizes)
    optimizer = th.optim.Adam(model.parameters(), lr=config.learning_rate)
    learner = ReinforceLearner(
        model,
        optimizer,
        ReinforceConfig(
            device=config.device,
            num_actions=num_actions,
            normalize_returns=True,
        ),
    )
    controller = StochasticController(model, num_actions=num_actions)
    experiment = OnPolicyExperiment(
        env,
        controller,
        learner,
        OnPolicyExperimentConfig(
            max_steps=config.steps,
            gamma=config.gamma,
            run_steps=0,
            log_dir=str(run_log_dir(config, "reinforce", DRLAB, seed)),
            experiment_name="drlab REINFORCE CartPole",
        ),
    )

    try:
        seconds = timed_train(experiment.run, config.show_progress)
        train_steps = env.step_count
    finally:
        env.close()

    result = evaluate_discrete_model(
        model,
        CARTPOLE,
        num_actions,
        config.eval_episodes,
        eval_seed(seed),
        config.device,
    )
    return RunResult(
        "reinforce",
        LEARNER_LABELS["reinforce"],
        DRLAB,
        CARTPOLE,
        seed,
        train_steps,
        seconds,
        result,
    )


def train_drlab_actor_critic(config: BenchmarkConfig, seed: int) -> RunResult:
    set_seed(seed)
    env = make_env(CARTPOLE, seed=seed, count_steps=True)
    obs_dim = flat_box_dim(env.observation_space)
    num_actions = discrete_action_count(env.action_space)

    model = make_mlp(obs_dim, num_actions + 1, config.hidden_sizes)
    optimizer = th.optim.Adam(model.parameters(), lr=config.learning_rate)
    learner = ActorCriticLearner(
        model,
        optimizer,
        ActorCriticConfig(
            device=config.device,
            num_actions=num_actions,
            gamma=config.gamma,
            value_targets=ValueTargets.TD,
            advantage_bootstrap=True,
            normalize_advantages=False,
            value_lambda=0.5,
        ),
    )
    controller = StochasticController(model, num_actions=num_actions)
    experiment = OnPolicyExperiment(
        env,
        controller,
        learner,
        OnPolicyExperimentConfig(
            max_steps=config.steps,
            gamma=config.gamma,
            run_steps=config.a2c_rollout_steps,
            log_dir=str(run_log_dir(config, "actor_critic", DRLAB, seed)),
            experiment_name="drlab Actor-Critic CartPole",
        ),
    )

    try:
        seconds = timed_train(experiment.run, config.show_progress)
        train_steps = env.step_count
    finally:
        env.close()

    result = evaluate_discrete_model(
        model,
        CARTPOLE,
        num_actions,
        config.eval_episodes,
        eval_seed(seed),
        config.device,
    )
    return RunResult(
        "actor_critic",
        LEARNER_LABELS["actor_critic"],
        DRLAB,
        CARTPOLE,
        seed,
        train_steps,
        seconds,
        result,
    )


def train_drlab_ppo(config: BenchmarkConfig, seed: int) -> RunResult:
    set_seed(seed)
    env = make_env(CARTPOLE, seed=seed, count_steps=True)
    obs_dim = flat_box_dim(env.observation_space)
    num_actions = discrete_action_count(env.action_space)

    model = make_mlp(obs_dim, num_actions + 1, config.hidden_sizes)
    optimizer = th.optim.Adam(model.parameters(), lr=config.learning_rate)
    learner = PPOLearner(
        model,
        optimizer,
        PPOConfig(
            device=config.device,
            num_actions=num_actions,
            gamma=config.gamma,
            value_targets=ValueTargets.TD,
            advantage_bootstrap=True,
            normalize_advantages=True,
            value_lambda=0.5,
            ppo_iterations=config.ppo_epochs,
            ppo_clipping=config.ppo_clip,
        ),
    )
    controller = StochasticController(model, num_actions=num_actions)
    experiment = OnPolicyExperiment(
        env,
        controller,
        learner,
        OnPolicyExperimentConfig(
            max_steps=config.steps,
            gamma=config.gamma,
            run_steps=config.ppo_rollout_steps,
            log_dir=str(run_log_dir(config, "ppo", DRLAB, seed)),
            experiment_name="drlab PPO CartPole",
        ),
    )

    try:
        seconds = timed_train(experiment.run, config.show_progress)
        train_steps = env.step_count
    finally:
        env.close()

    result = evaluate_discrete_model(
        model,
        CARTPOLE,
        num_actions,
        config.eval_episodes,
        eval_seed(seed),
        config.device,
    )
    return RunResult("ppo", LEARNER_LABELS["ppo"], DRLAB, CARTPOLE, seed, train_steps, seconds, result)


def train_drlab_sac(config: BenchmarkConfig, seed: int) -> RunResult:
    set_seed(seed)
    env = make_env(PENDULUM, seed=seed, count_steps=True)
    obs_dim = flat_box_dim(env.observation_space)
    action_dim = flat_box_dim(env.action_space)

    actor = make_mlp(obs_dim, 2 * action_dim, config.hidden_sizes)
    critic1 = QNetwork(obs_dim, action_dim, config.hidden_sizes)
    critic2 = QNetwork(obs_dim, action_dim, config.hidden_sizes)

    learner = SACLearner(
        actor=actor,
        critic1=critic1,
        critic2=critic2,
        actor_optimizer=th.optim.Adam(actor.parameters(), lr=3e-4),
        critic1_optimizer=th.optim.Adam(critic1.parameters(), lr=3e-4),
        critic2_optimizer=th.optim.Adam(critic2.parameters(), lr=3e-4),
        config=SACConfig(
            device=config.device,
            gamma=config.gamma,
            action_shape=env.action_space.shape,
            soft_target_update_param=config.sac_tau,
        ),
    )
    controller = GaussianController(actor, action_dim=action_dim)
    experiment = OffPolicyExperiment(
        env,
        controller,
        learner,
        OffPolicyExperimentConfig(
            max_steps=config.steps,
            gamma=config.gamma,
            run_steps=1,
            batch_size=config.sac_batch_size,
            replay_buffer_size=config.sac_replay_size,
            use_last_episode=False,
            grad_repeats=1,
            log_dir=str(run_log_dir(config, "sac", DRLAB, seed)),
            experiment_name="drlab SAC Pendulum",
        ),
    )

    try:
        seconds = timed_train(experiment.run, config.show_progress)
        train_steps = env.step_count
    finally:
        env.close()

    result = evaluate_sac_actor(
        actor,
        PENDULUM,
        action_dim,
        config.eval_episodes,
        eval_seed(seed),
        config.device,
    )
    return RunResult("sac", LEARNER_LABELS["sac"], DRLAB, PENDULUM, seed, train_steps, seconds, result)


def sb3_policy_kwargs(config: BenchmarkConfig) -> dict[str, Any]:
    return {"net_arch": list(config.hidden_sizes)}


def train_sb3_dqn(config: BenchmarkConfig, seed: int) -> RunResult:
    from stable_baselines3 import DQN

    set_seed(seed)
    env = make_env(CARTPOLE, seed=seed)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        buffer_size=config.dqn_replay_size,
        learning_starts=config.dqn_batch_size,
        batch_size=config.dqn_batch_size,
        gamma=config.gamma,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=config.dqn_target_update_interval,
        exploration_fraction=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=sb3_policy_kwargs(config),
        tensorboard_log=str(run_log_dir(config, "dqn", SB3, seed)),
        seed=seed,
        verbose=0,
        device=config.device,
    )

    try:
        seconds = timed_train(
            lambda: model.learn(total_timesteps=config.steps, tb_log_name="dqn"),
            config.show_progress,
        )
        train_steps = int(model.num_timesteps)
    finally:
        close_sb3_env(model)

    result = evaluate_sb3_model(model, CARTPOLE, config.eval_episodes, eval_seed(seed))
    return RunResult("dqn", LEARNER_LABELS["dqn"], SB3, CARTPOLE, seed, train_steps, seconds, result)


def train_sb3_a2c(config: BenchmarkConfig, seed: int) -> RunResult:
    from stable_baselines3 import A2C

    set_seed(seed)
    env = make_env(CARTPOLE, seed=seed)
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.a2c_rollout_steps,
        gamma=config.gamma,
        gae_lambda=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=1.0,
        use_rms_prop=False,
        normalize_advantage=False,
        policy_kwargs=sb3_policy_kwargs(config),
        tensorboard_log=str(run_log_dir(config, "actor_critic", SB3, seed)),
        seed=seed,
        verbose=0,
        device=config.device,
    )

    try:
        seconds = timed_train(
            lambda: model.learn(total_timesteps=config.steps, tb_log_name="a2c"),
            config.show_progress,
        )
        train_steps = int(model.num_timesteps)
    finally:
        close_sb3_env(model)

    result = evaluate_sb3_model(model, CARTPOLE, config.eval_episodes, eval_seed(seed))
    return RunResult(
        "actor_critic",
        LEARNER_LABELS["actor_critic"],
        SB3,
        CARTPOLE,
        seed,
        train_steps,
        seconds,
        result,
    )


def train_sb3_ppo(config: BenchmarkConfig, seed: int) -> RunResult:
    from stable_baselines3 import PPO

    set_seed(seed)
    env = make_env(CARTPOLE, seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        n_steps=config.ppo_rollout_steps,
        batch_size=config.ppo_rollout_steps,
        n_epochs=config.ppo_epochs,
        gamma=config.gamma,
        gae_lambda=1.0,
        clip_range=config.ppo_clip,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=1.0,
        policy_kwargs=sb3_policy_kwargs(config),
        tensorboard_log=str(run_log_dir(config, "ppo", SB3, seed)),
        seed=seed,
        verbose=0,
        device=config.device,
    )

    try:
        seconds = timed_train(
            lambda: model.learn(total_timesteps=config.steps, tb_log_name="ppo"),
            config.show_progress,
        )
        train_steps = int(model.num_timesteps)
    finally:
        close_sb3_env(model)

    result = evaluate_sb3_model(model, CARTPOLE, config.eval_episodes, eval_seed(seed))
    return RunResult("ppo", LEARNER_LABELS["ppo"], SB3, CARTPOLE, seed, train_steps, seconds, result)


def train_sb3_sac(config: BenchmarkConfig, seed: int) -> RunResult:
    from stable_baselines3 import SAC

    set_seed(seed)
    env = make_env(PENDULUM, seed=seed)
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=config.sac_replay_size,
        learning_starts=config.sac_batch_size,
        batch_size=config.sac_batch_size,
        tau=config.sac_tau,
        gamma=config.gamma,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=sb3_policy_kwargs(config),
        tensorboard_log=str(run_log_dir(config, "sac", SB3, seed)),
        seed=seed,
        verbose=0,
        device=config.device,
    )

    try:
        seconds = timed_train(
            lambda: model.learn(total_timesteps=config.steps, tb_log_name="sac"),
            config.show_progress,
        )
        train_steps = int(model.num_timesteps)
    finally:
        close_sb3_env(model)

    result = evaluate_sb3_model(model, PENDULUM, config.eval_episodes, eval_seed(seed))
    return RunResult("sac", LEARNER_LABELS["sac"], SB3, PENDULUM, seed, train_steps, seconds, result)


DRLAB_TRAINERS: dict[str, Callable[[BenchmarkConfig, int], RunResult]] = {
    "dqn": train_drlab_dqn,
    "reinforce": train_drlab_reinforce,
    "actor_critic": train_drlab_actor_critic,
    "ppo": train_drlab_ppo,
    "sac": train_drlab_sac,
}

SB3_TRAINERS: dict[str, Callable[[BenchmarkConfig, int], RunResult]] = {
    "dqn": train_sb3_dqn,
    "actor_critic": train_sb3_a2c,
    "ppo": train_sb3_ppo,
    "sac": train_sb3_sac,
}


def run_benchmarks(
    config: BenchmarkConfig,
    learners: Sequence[str],
    seeds: Sequence[int],
    include_sb3: bool,
) -> list[RunResult]:
    results: list[RunResult] = []
    total_runs = len(seeds) * (
        len(learners) + sum(1 for learner in learners if include_sb3 and learner in SB3_TRAINERS)
    )
    run_index = 1

    for learner_key in learners:
        learner_label = LEARNER_LABELS[learner_key]
        for seed in seeds:
            print(f"[{run_index}/{total_runs}] Training {DRLAB} {learner_label} seed={seed}")
            results.append(DRLAB_TRAINERS[learner_key](config, seed))
            run_index += 1

            if include_sb3 and learner_key in SB3_TRAINERS:
                print(f"[{run_index}/{total_runs}] Training {SB3} {learner_label} seed={seed}")
                results.append(SB3_TRAINERS[learner_key](config, seed))
                run_index += 1

    return results


def summarize_results(results: Sequence[RunResult], learners: Sequence[str]) -> list[SummaryRow]:
    rows: list[SummaryRow] = []

    for learner_key in learners:
        for framework in (DRLAB, SB3):
            runs = [
                result
                for result in results
                if result.learner_key == learner_key and result.framework == framework
            ]
            if not runs:
                continue

            returns = [episode_return for run in runs for episode_return in run.eval.returns]
            eval_mean = float(np.mean(returns))
            eval_std = float(np.std(returns))
            train_seconds = float(np.mean([run.train_seconds for run in runs]))
            steps_per_second = float(np.mean([run.steps_per_second for run in runs]))

            rows.append(
                SummaryRow(
                    learner_key=learner_key,
                    learner_label=runs[0].learner_label,
                    framework=framework,
                    env_id=runs[0].env.env_id,
                    seeds=len(runs),
                    eval_mean=eval_mean,
                    eval_std=eval_std,
                    accuracy_pct=runs[0].env.accuracy_pct(eval_mean),
                    train_seconds=train_seconds,
                    steps_per_second=steps_per_second,
                )
            )

    sb3_rates = {
        row.learner_key: row.steps_per_second
        for row in rows
        if row.framework == SB3 and row.steps_per_second > 0.0
    }
    for row in rows:
        if row.framework == SB3:
            row.speedup_vs_sb3 = 1.0
        elif row.learner_key in sb3_rates:
            row.speedup_vs_sb3 = row.steps_per_second / sb3_rates[row.learner_key]

    return rows


def format_rate(value: float) -> str:
    if value >= 100.0:
        return f"{value:.0f}"
    if value >= 10.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def format_speedup(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}x"


def render_table(rows: Sequence[SummaryRow]) -> str:
    headers = [
        "Learner",
        "Library",
        "Env",
        "Seeds",
        "Eval return",
        "Accuracy",
        "Train s",
        "Steps/s",
        "Speedup",
    ]
    table_rows = [
        [
            row.learner_label,
            row.framework,
            row.env_id,
            str(row.seeds),
            f"{row.eval_mean:.2f} +/- {row.eval_std:.2f}",
            f"{row.accuracy_pct:.1f}%",
            f"{row.train_seconds:.2f}",
            format_rate(row.steps_per_second),
            format_speedup(row.speedup_vs_sb3),
        ]
        for row in rows
    ]
    widths = [
        max(len(str(row[col])) for row in [headers, *table_rows])
        for col in range(len(headers))
    ]

    def render_row(row: Sequence[str]) -> str:
        return "  ".join(str(value).ljust(width) for value, width in zip(row, widths))

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render_row(headers), separator, *[render_row(row) for row in table_rows]])


def write_results_csv(path: Path, rows: Sequence[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "learner",
                "library",
                "environment",
                "seeds",
                "eval_return_mean",
                "eval_return_std",
                "accuracy_pct",
                "train_seconds_mean",
                "steps_per_second_mean",
                "speedup_vs_sb3",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "learner": row.learner_label,
                    "library": row.framework,
                    "environment": row.env_id,
                    "seeds": row.seeds,
                    "eval_return_mean": row.eval_mean,
                    "eval_return_std": row.eval_std,
                    "accuracy_pct": row.accuracy_pct,
                    "train_seconds_mean": row.train_seconds,
                    "steps_per_second_mean": row.steps_per_second,
                    "speedup_vs_sb3": row.speedup_vs_sb3,
                }
            )


def print_metric_notes(include_sb3: bool, learners: Sequence[str]) -> None:
    print("\nMetric notes:")
    print("- Accuracy is normalized deterministic evaluation return.")
    print("- CartPole accuracy uses 0..500 return; Pendulum uses theoretical worst return..0.")
    if include_sb3:
        print("- Speedup is mean training transitions/second relative to the matching SB3 row.")
    if "reinforce" in learners:
        print("- REINFORCE has no direct SB3 implementation, so its speedup is blank.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.torch_threads > 0:
        th.set_num_threads(args.torch_threads)

    learners = unique_learners(args.learners)
    seeds = selected_seeds(args)
    include_sb3 = not args.skip_sb3 and any(learner in SB3_TRAINERS for learner in learners)
    if include_sb3:
        require_stable_baselines3()

    config = BenchmarkConfig(
        steps=args.steps,
        eval_episodes=args.eval_episodes,
        log_dir=Path(args.log_dir),
        device=args.device,
        show_progress=args.progress,
    )

    print(
        f"Benchmarking {len(learners)} learner(s), {len(seeds)} seed(s), "
        f"{config.steps} requested env steps per run."
    )
    if not args.progress:
        print("Progress bars hidden; pass --progress to show them.")

    results = run_benchmarks(config, learners, seeds, include_sb3)
    rows = summarize_results(results, learners)

    print("\nResults:")
    print(render_table(rows))
    print_metric_notes(include_sb3, learners)

    if not args.no_save:
        results_csv = Path(args.results_csv) if args.results_csv else config.log_dir / "results.csv"
        write_results_csv(results_csv, rows)
        print(f"\nSaved aggregate CSV: {results_csv}")

    print(f"TensorBoard logs: {config.log_dir}")


if __name__ == "__main__":
    main()
