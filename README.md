# drlab

`drlab` is a small deep reinforcement learning package for research code and
experiments. It provides reusable building blocks for Gymnasium environments:

- DQN, REINFORCE, actor-critic, and PPO learners
- greedy, epsilon-greedy, and stochastic controllers
- a transition runner for collecting environment interaction
- replay buffer and transition batch utilities
- lightweight experiment wrappers with TensorBoard logging

The package is designed around small, composable pieces: a PyTorch model,
controller, runner, learner, and optionally an experiment wrapper.

## Installation

The recommended development setup uses
[`uv`](https://docs.astral.sh/uv/) to create a private `.venv`, install the
package in editable mode, and keep dependencies locked.

Install `uv` once:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then, from the repository root:

```bash
uv sync --extra experiments --extra dev
```

Run commands through the managed environment:

```bash
uv run python -m unittest discover -v
uv run python -c "import drlab; print(drlab.__version__)"
```

Or activate the private environment manually:

```bash
source .venv/bin/activate
python -m unittest discover -v
```

The package depends on PyTorch `>=2.0,<3`. For a specific CUDA build, install
the matching PyTorch wheel for your machine before syncing the rest of the
environment, for example:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv sync --extra experiments --extra dev
```

Commit `pyproject.toml` and `uv.lock`, but do not commit `.venv/`.

For a plain `pip` install, create and activate a virtual environment first.
From the repository root:

```bash
python -m pip install -e .
```

For experiment and development dependencies:

```bash
python -m pip install -e ".[experiments,dev]"
```

## Package Overview

Public classes are available from the package root:

```python
from drlab import (
    ActorCriticConfig,
    ActorCriticLearner,
    OnPolicyExperiment,
    OnPolicyExperimentConfig,
    Controller,
    DQNConfig,
    DQNLearner,
    PPOConfig,
    PPOLearner,
    ReinforceConfig,
    ReinforceLearner,
    OffPolicyExperiment,
    OffPolicyExperimentConfig,
    EpsilonGreedyController,
    GreedyController,
    ReplayBuffer,
    Runner,
    StochasticController,
    TransitionBatch,
)
```

They can also be imported from their subpackages:

| Subpackage | Exports | Purpose |
| --- | --- | --- |
| `drlab.learners` | `DQNLearner`, `DQNConfig`, `ReinforceLearner`, `ActorCriticLearner`, `PPOLearner` and configs | Update PyTorch models from transition batches. |
| `drlab.controllers` | `Controller`, `GreedyController`, `EpsilonGreedyController`, `StochasticController` | Convert model outputs into environment actions. |
| `drlab.runners` | `Runner` | Collect transitions from a Gymnasium environment. |
| `drlab.replay` | `ReplayBuffer`, `TransitionBatch` | Store, sample, move, and concatenate transitions. |
| `drlab.experiments` | `OffPolicyExperiment`, `OffPolicyExperimentConfig`, `OnPolicyExperiment`, `OnPolicyExperimentConfig` | Run training loops with logging and progress bars. |

## Implemented Algorithms

For a detailed explanation of each algorithm, training flow, and configuration
parameter, see [RL Algorithms and Parameters](docs/rl_algorithms.md).

| Algorithm | Type | Implementation Summary |
| --- | --- | --- |
| DQN | Off-policy value-based RL | Trains a Q-network with one-step TD targets from `(state, action, reward, done, next_state)` batches. It supports replay-buffer training through `OffPolicyExperiment`, target networks, Double DQN action selection, hard or soft target-network updates, gradient clipping, configurable discounting, and custom regularizers. |
| REINFORCE | On-policy policy-gradient RL | Trains a stochastic policy directly from discounted returns, with optional return normalization, entropy regularization, gradient clipping, and custom regularizers. |
| Actor-Critic | On-policy policy-gradient RL | Trains a shared policy/value network from transition batches and returns. The policy head is optimized with advantage-weighted log probabilities, while the value head can use TD targets or full returns. It supports bootstrapped advantages, optional baseline subtraction, advantage normalization, entropy regularization with annealing, gradient clipping, and custom regularizers. |
| PPO | On-policy policy-gradient RL | Reuses an actor-critic rollout batch for multiple clipped-ratio optimization epochs through `PPOLearner` and `PPOConfig`. |

The package also includes reusable action-selection controllers:

- `GreedyController`: deterministic argmax action selection from model scores.
- `EpsilonGreedyController`: epsilon-greedy exploration with linear annealing.
- `StochasticController`: samples actions from softmax probabilities.

## Model Output Convention

Controllers and learners expect the model output to use a shared layout:

- DQN models should output at least `num_actions` columns. The first
  `num_actions` columns are treated as action scores.
- REINFORCE models should output at least `num_actions` policy-logit columns.
- Actor-critic and PPO models should output at least `num_actions + 1` columns. The
  first `num_actions` columns are policy logits, and the next column is the
  value estimate.

## Quick DQN Example

```python
import gymnasium as gym
import torch as th

from drlab import (
    DQNConfig,
    DQNLearner,
    OffPolicyExperiment,
    OffPolicyExperimentConfig,
    EpsilonGreedyController,
    GreedyController,
)

env = gym.make("CartPole-v1")

model = th.nn.Sequential(
    th.nn.Linear(4, 64),
    th.nn.ReLU(),
    th.nn.Linear(64, 2),
)
optimizer = th.optim.Adam(model.parameters(), lr=1e-3)

learner = DQNLearner(model, optimizer, DQNConfig(num_actions=2))
controller = EpsilonGreedyController(
    GreedyController(model, num_actions=2),
    num_actions=2,
    max_eps=1.0,
    min_eps=0.05,
    anneal_steps=10_000,
)

experiment = OffPolicyExperiment(
    env,
    controller,
    learner,
    OffPolicyExperimentConfig(
        max_steps=20_000,
        run_steps=1,
        batch_size=128,
        log_dir="runs/cartpole_dqn",
    ),
)
experiment.run()
```

## Core Components

### Learners

`DQNLearner` trains a Q-network from `(rewards, dones, states, actions, next_states)`.
Its config supports target networks, double Q-learning, hard or soft target
updates, gradient clipping, discounting, and custom regularizers.

```python
from drlab.learners import DQNConfig, DQNLearner
```

`ActorCriticLearner` trains a policy/value network from transition batches with
returns. Its config supports TD or return-based value targets, bootstrapped
advantages, entropy regularization, advantage normalization, and custom
regularizers.

```python
from drlab.learners import ActorCriticLearner, ActorCriticConfig
```

`ReinforceLearner` and `PPOLearner` provide separate REINFORCE and PPO
implementations:

```python
from drlab.learners import ReinforceLearner, ReinforceConfig, PPOLearner, PPOConfig
```

### Controllers

Controllers wrap a PyTorch model and expose:

```python
action = controller.choose(obs)
probs = controller.probabilities(obs)
```

Available controllers:

- `GreedyController`: selects the highest-scoring action.
- `EpsilonGreedyController`: wraps another controller and adds annealed random
  exploration.
- `StochasticController`: samples actions from softmax probabilities.

### Runner

`Runner` steps through a Gymnasium environment with a controller and returns:

```python
batch, ep_returns, ep_lengths, last_episode = runner.run(num_steps)
```

`num_steps <= 0` collects one complete episode. Positive values collect up to
that many transitions. The returned `batch` is a `TransitionBatch`.

### Replay

`TransitionBatch` stores tensors for:

- `states`
- `actions`
- `rewards`
- `dones`
- `next_states`
- `returns`

It provides `.to(device)` and `.cat(other)` helpers.

`ReplayBuffer` stores fixed-capacity NumPy arrays and returns sampled or full
data as `TransitionBatch` instances:

```python
buffer = ReplayBuffer(capacity=10_000, obs_shape=env.observation_space.shape)
batch = buffer.sample(128)
all_data = buffer.get_all()
```

### Experiments

Experiment wrappers combine an environment, controller, learner, runner, replay
buffer behavior, progress bar, and TensorBoard logging.

```python
from drlab.experiments import (
    OnPolicyExperiment,
    OnPolicyExperimentConfig,
    OffPolicyExperiment,
    OffPolicyExperimentConfig,
)
```

Use `OffPolicyExperiment` for off-policy DQN training and `OnPolicyExperiment` for
on-policy actor-critic training.

## Development

Install development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
python -m unittest discover -v
```

Build a wheel:

```bash
python -m build --wheel
```
