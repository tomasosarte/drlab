# drlab

`drlab` is a small deep reinforcement learning library for training neural
network agents in Gymnasium environments. Its purpose is to provide simple,
research-friendly building blocks for reinforcement learning experiments,
especially when the environment or training setup is still being explored.

The library is built around PyTorch models, Gymnasium environments, reusable
controllers, replay buffers, runners, learners, and lightweight experiment
loops. It is intended for research code and prototypes, not as a large
production RL framework.

## Installation

The intended development setup uses [`uv`](https://docs.astral.sh/uv/). From the
repository root:

```bash
uv sync --extra experiments --extra dev
```

Run commands through the managed environment:

```bash
uv run python -m pytest
uv run python examples/dqn_cartpole.py
```

You can also activate the local virtual environment created by `uv`:

```bash
source .venv/bin/activate
python -m pytest
```

If you prefer a normal `venv` and `pip` workflow:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For experiment and development dependencies:

```bash
python -m pip install -e ".[experiments,dev]"
```

`drlab` depends on PyTorch, NumPy, Gymnasium, tqdm, and TensorBoard. If you need
a specific CUDA build of PyTorch, install the matching PyTorch wheel before
installing the package dependencies.

## Library Overview

Most public classes can be imported directly from `drlab`:

```python
from drlab import DQNConfig, DQNLearner, OffPolicyExperiment
```

### Off-Policy Learning

Off-policy learners train from transition batches that may come from a replay
buffer instead of only the most recent rollout. `OffPolicyExperiment` collects
environment transitions with a `Runner`, stores them in a `ReplayBuffer`, samples
minibatches, and calls the learner update.

Implemented off-policy learners:

- `DQNLearner`: value-based learning for discrete action spaces. It supports
  replay-buffer training, target networks, Double DQN, hard or soft target
  updates, gradient clipping, and custom regularizers.
- `SACLearner`: Soft Actor-Critic style learning for continuous action spaces.
  It uses an actor, two critics, target critics, entropy tuning, and replay
  buffer training.

### On-Policy Learning

On-policy learners train from the data collected by the current policy.
`OnPolicyExperiment` collects rollouts, computes returns when needed, and trains
the learner directly on the fresh batch.

Implemented on-policy learners:

- `ReinforceLearner`: policy-gradient learning from discounted returns.
- `ActorCriticLearner`: policy and value learning from shared model outputs.
- `PPOLearner`: clipped policy optimization over rollout batches.

### Supporting Components

Controllers turn model outputs into environment actions:

- `GreedyController`, `EpsilonGreedyController`, and `StochasticController`
  support discrete action spaces.
- `GaussianController` supports continuous action spaces by sampling bounded
  actions from model-predicted Gaussian parameters.

`Runner` interacts with Gymnasium environments and returns transition batches,
episode returns, episode lengths, and optionally the last completed episode.

`ReplayBuffer` stores transitions for off-policy learning. It supports both
discrete and continuous action spaces through the configured action shape and
action type.

`TransitionBatch` is the tensor container passed between runners, replay
buffers, experiments, and learners.

## Simple Usage

This example trains a DQN agent on `CartPole-v1`:

```python
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
    anneal_steps=10_000,
)

experiment = OffPolicyExperiment(
    env,
    controller,
    learner,
    OffPolicyExperimentConfig(
        max_steps=10_000,
        run_steps=1,
        batch_size=64,
        warmup_steps=1_000,
        log_dir="runs/examples/dqn_cartpole",
    ),
)

experiment.run()
env.close()
```

More complete scripts are available in the [`examples/`](examples/) folder,
including DQN, REINFORCE, actor-critic, PPO, SAC, and a comparison with
Stable-Baselines3.
