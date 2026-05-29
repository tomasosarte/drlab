# drlab

`drlab` is a small deep reinforcement learning package for research code and
experiments. It provides reusable building blocks for Gymnasium environments:

- DQN and actor-critic learners
- greedy, epsilon-greedy, and stochastic controllers
- a transition runner for collecting environment interaction
- replay buffer and transition batch utilities
- lightweight experiment wrappers with TensorBoard logging

The package is designed around small, composable pieces: a PyTorch model,
controller, runner, learner, and optionally an experiment wrapper.

## Installation

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
    ActorCritic,
    ActorCriticConfig,
    ActorCriticExperiment,
    ActorCriticExperimentConfig,
    Controller,
    DQN,
    DQNConfig,
    DQNExperiment,
    DQNExperimentConfig,
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
| `drlab.learners` | `DQN`, `DQNConfig`, `ActorCritic`, `ActorCriticConfig` | Update PyTorch models from transition batches. |
| `drlab.controllers` | `Controller`, `GreedyController`, `EpsilonGreedyController`, `StochasticController` | Convert model outputs into environment actions. |
| `drlab.runners` | `Runner` | Collect transitions from a Gymnasium environment. |
| `drlab.replay` | `ReplayBuffer`, `TransitionBatch` | Store, sample, move, and concatenate transitions. |
| `drlab.experiments` | `DQNExperiment`, `DQNExperimentConfig`, `ActorCriticExperiment`, `ActorCriticExperimentConfig` | Run training loops with logging and progress bars. |

## Model Output Convention

Controllers and learners expect the model output to use a shared layout:

- DQN models should output at least `num_actions` columns. The first
  `num_actions` columns are treated as action scores.
- Actor-critic models should output at least `num_actions + 1` columns. The
  first `num_actions` columns are policy logits, and the next column is the
  value estimate.

## Quick DQN Example

```python
import gymnasium as gym
import torch as th

from drlab import (
    DQN,
    DQNConfig,
    DQNExperiment,
    DQNExperimentConfig,
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

learner = DQN(model, optimizer, DQNConfig(num_actions=2))
controller = EpsilonGreedyController(
    GreedyController(model, num_actions=2),
    num_actions=2,
    max_eps=1.0,
    min_eps=0.05,
    anneal_steps=10_000,
)

experiment = DQNExperiment(
    env,
    controller,
    learner,
    DQNExperimentConfig(
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

`DQN` trains a Q-network from `(rewards, dones, states, actions, next_states)`.
Its config supports target networks, double Q-learning, hard or soft target
updates, gradient clipping, discounting, and custom regularizers.

```python
from drlab.learners import DQN, DQNConfig
```

`ActorCritic` trains a policy/value network from transition batches with
returns. Its config supports TD or return-based value targets, bootstrapped
advantages, PPO-style clipping, entropy regularization, advantage
normalization, and custom regularizers.

```python
from drlab.learners import ActorCritic, ActorCriticConfig
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
    ActorCriticExperiment,
    ActorCriticExperimentConfig,
    DQNExperiment,
    DQNExperimentConfig,
)
```

Use `DQNExperiment` for off-policy DQN training and `ActorCriticExperiment` for
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
