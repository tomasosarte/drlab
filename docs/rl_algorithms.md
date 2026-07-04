# RL Algorithms and Parameters

This document explains the reinforcement learning algorithms implemented in
`drlab` and the most important parameters that affect training behavior.

The package currently implements:

- Deep Q-Network (DQN), an off-policy value-based algorithm for discrete
  action spaces.
- Actor-critic, an on-policy policy-gradient algorithm with a shared policy
  and value model.

Both learners operate on `TransitionBatch` objects collected by `Runner`, and
both expect a PyTorch model that receives a batch of observations.

## Shared Conventions

### Transition Data

Learners consume tensors with the following shapes:

| Field | Shape | Type | Meaning |
| --- | --- | --- | --- |
| `states` | `[B, *obs_shape]` | `float32` | Current observations. |
| `actions` | `[B, 1]` | `int64` | Discrete action indices. |
| `rewards` | `[B, 1]` | `float32` | Immediate rewards. |
| `dones` | `[B, 1]` | `bool` | Episode termination or truncation flags. |
| `next_states` | `[B, *obs_shape]` | `float32` | Observations after each action. |
| `returns` | `[B, 1]` | `float32` | Discounted returns, when requested from the runner. |

`TransitionBatch.to(device)` moves every tensor to a device, and
`TransitionBatch.cat(other)` concatenates two batches along the batch
dimension.

### Model Output Layout

`num_actions` must match the number of discrete actions in the environment.

| Learner or controller | Required model output |
| --- | --- |
| DQN and greedy controllers | At least `num_actions` columns. The first columns are action scores or Q-values. |
| Actor-critic | At least `num_actions + 1` columns. The first columns are policy logits, and column `num_actions` is the value estimate. |
| Stochastic controller | At least `num_actions` columns. The first columns are logits that are converted with `softmax`. |
| Gaussian controller | At least `2 * action_dim` columns. The first columns are means and the next columns are log standard deviations. |

For actor-critic, keep the value column even when value loss is disabled. Most
configurations use it for bootstrapped advantages, TD value targets, or both.

### Discount Factors

There are two places where `gamma` can appear:

- Learner configs use `gamma` inside the optimization target.
- Experiment configs pass `gamma` to `Runner` for discounted return
  calculation.

For actor-critic settings that use full returns, keep these values aligned.
DQN does not use `TransitionBatch.returns`, so the experiment-level `gamma` is
mostly relevant to runner bookkeeping for DQN.

## Deep Q-Network

`DQNLearner` is an off-policy value-based learner. It trains a Q-network from sampled
transitions and is usually paired with `EpsilonGreedyController`,
`OffPolicyExperiment`, and `ReplayBuffer`.

### Update Rule

For each transition `(state, action, reward, done, next_state)`, DQN computes a
one-step TD target:

```text
target = reward + gamma * (1 - done) * next_value
```

The current estimate is:

```text
prediction = Q_online(state, action)
```

The learner applies `criterion(prediction, target)` and then adds any configured
regularization losses.

### Target Network

When `use_target_model=True`, the learner creates a frozen copy of the online
model. The target model is used to evaluate future Q-values and is updated
after training steps.

Two target update modes are supported:

| Mode | Behavior |
| --- | --- |
| `soft` | Every learner update blends target parameters toward online parameters using `tau = soft_target_update_param`. |
| `hard` | Every `target_update_interval` learner updates, copies the entire online model into the target model. |

With soft updates, the target parameters are updated as:

```text
target_param = (1 - tau) * target_param + tau * online_param
```

### Double DQN

When `double_q=True`, action selection and action evaluation are separated:

```text
best_action = argmax_a Q_online(next_state, a)
next_value = Q_target(next_state, best_action)
```

This reduces overestimation compared with selecting and evaluating the next
action using the same network. Because it needs a target network,
`double_q=True` requires `use_target_model=True`.

When `double_q=False`, the next value is the maximum Q-value from the target
network if one is enabled, otherwise from the online network:

```text
next_value = max_a Q(next_state, a)
```

### DQNConfig Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `criterion` | `th.nn.MSELoss()` | Loss function for predicted Q-values against TD targets. It should accept tensors shaped `[B, 1]`. |
| `regularizers` | `[]` | Optional callables that add extra losses. Each receives `(model, rewards, dones, states, actions, next_states)`. |
| `reg_lams` | `[]` | Scalar weights for `regularizers`. Must have the same length as `regularizers`. |
| `clip_grad` | `True` | Enables gradient norm clipping after backpropagation. |
| `grad_norm_clip` | `1.0` | Maximum gradient norm used when `clip_grad=True`. |
| `gamma` | `0.99` | Discount factor used in the TD target. |
| `double_q` | `True` | Enables Double DQN action selection. Requires a target model. |
| `device` | `"cpu"` | Device for the model and learner tensors. |
| `use_target_model` | `True` | Creates and uses a target network for next-state value evaluation. |
| `target_update` | `"soft"` | Target network update mode. Must be `"soft"` or `"hard"`. |
| `target_update_interval` | `100` | Number of learner updates between hard target copies. Must be positive for hard updates. |
| `soft_target_update_param` | `0.1` | Soft update coefficient `tau`. Must be in `(0, 1]` for soft updates. |
| `num_actions` | `2` | Number of action columns read from the model output. |

### DQN Training Flow

`DQNLearner.train(...)` performs one optimizer step:

1. Switches the online model to train mode.
2. Computes the one-step TD target.
3. Reads `Q(state, action)` from the online model.
4. Computes TD loss plus regularization losses.
5. Clears gradients, backpropagates, optionally clips gradients, and steps the
   optimizer.
6. Updates the target model according to `target_update`.
7. Returns the scalar loss as a Python `float`.

### OffPolicyExperiment Parameters

`OffPolicyExperiment` provides a full training loop around `DQNLearner`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_steps` | Required | Total environment transitions to collect before stopping. |
| `gamma` | `0.99` | Discount used by `Runner` if it computes returns. DQN itself uses `DQNConfig.gamma`. |
| `run_steps` | `0` | Number of transitions collected per runner call. `0` or less means collect one full episode. |
| `log_dir` | `"runs/off_policy_experiment"` | TensorBoard log directory. |
| `experiment_name` | `"OffPolicyExperiment"` | Progress bar label. |
| `use_replay` | `True` | Stores transitions in a replay buffer and samples minibatches. |
| `replay_buffer_size` | `10_000` | Maximum number of transitions retained by the replay buffer. |
| `batch_size` | `128` | Minibatch size and minimum buffer size before learning starts. |
| `use_last_episode` | `True` | When replay is enabled, includes the latest complete episode in the minibatch when available. |
| `grad_repeats` | `1` | Number of learner updates per collected batch. The logged loss is averaged over repeats. |
| `step_callback` | `None` | Optional callable invoked at step `0` and then on configured step intervals. |
| `step_callback_interval` | `None` | Step spacing for `step_callback`. Required when a callback is provided. |

Important behavior:

- Learning does not start until the replay buffer contains at least
  `batch_size` transitions.
- `ReplayBuffer.sample(batch_size)` samples uniformly with replacement and
  returns fewer than `batch_size` transitions only when the buffer is smaller.
  `OffPolicyExperiment` avoids this by waiting for `buffer.size >= batch_size`.
- With `use_replay=False`, the internal buffer capacity is `batch_size`.
  Ensure a collected runner batch is not larger than `batch_size`.
- With `use_last_episode=True`, a full latest episode can dominate the
  minibatch if the episode length is greater than or equal to `batch_size`.

### Typical DQN Settings

For discrete-control tasks such as CartPole, common starting points are:

- `double_q=True`
- `use_target_model=True`
- `target_update="soft"` with `soft_target_update_param` around `0.01` to
  `0.1`, or `target_update="hard"` with a larger interval.
- `batch_size` from `64` to `256`.
- `replay_buffer_size` much larger than `batch_size`.
- `EpsilonGreedyController` with exploration annealed from high epsilon to a
  small nonzero final epsilon.

## Actor-Critic

`ActorCriticLearner` is an on-policy policy-gradient learner. It expects one
model that emits both policy logits and a scalar value estimate:

```text
output[:, :num_actions]              -> policy logits
output[:, num_actions:num_actions+1] -> value estimate
```

It is usually paired with `StochasticController` and `OnPolicyExperiment`.

### Policy Loss

The model logits are converted to action probabilities with `softmax`. The
learner gathers the probability of the action that was actually taken:

```text
pi_action = pi(action | state)
```

The policy loss is:

```text
policy_loss = -mean(log(pi_action) * advantage)
```

The advantage is detached before the policy gradient is computed.

### Advantage Estimates

The advantage source is controlled by `advantage_bootstrap`:

| Setting | Advantage source |
| --- | --- |
| `advantage_bootstrap=True` | `reward + gamma * (1 - done) * V(next_state)` |
| `advantage_bootstrap=False` | `returns` from the runner |

When `use_bias=True`, the current value estimate is subtracted:

```text
advantage = advantage_source - V(state)
```

This turns the value function into a baseline and can reduce variance. The
subtracted value is detached in the policy loss, so the policy objective does
not directly backpropagate through the baseline.

When `normalize_advantages=True`, advantages are centered and scaled inside the
batch:

```text
advantage = (advantage - mean(advantage)) / (std(advantage) + 1e-8)
```

### Value Loss

When `use_bias=True`, the learner also trains the value head. The value target
is controlled by `value_targets`:

| Setting | Value target |
| --- | --- |
| `"td"` | `reward + gamma * (1 - done) * V(next_state)` |
| `"returns"` | Discounted returns from the runner. |

The value loss is:

```text
value_lambda * value_criterion(V(state), value_target)
```

When `use_bias=False`, no value loss is added.

### PPO Learner

PPO is implemented separately by `PPOLearner` and `PPOConfig`. It uses the
same policy/value model layout and advantage/value logic as
`ActorCriticLearner`, then reuses the collected batch for `ppo_iterations`
optimization epochs. Before the first update, it stores the old action
probabilities. Each epoch uses a ratio objective with clipping:

```text
ratio = pi_new(action | state) / pi_old(action | state)
clipped_ratio = clamp(ratio, 1 - ppo_clipping, 1 + ppo_clipping)
loss = -mean(min(ratio * advantage, clipped_ratio * advantage))
```

The implementation does not add generalized advantage estimation, minibatch
shuffling, or a separate old policy network.

### Entropy Regularization

When `use_entropy=True`, the learner adds an entropy bonus through a negative
entropy loss:

```text
entropy_loss = -entropy_lambda * mean(policy_entropy)
```

`entropy_lambda` is linearly annealed from `entropy_max_lambda` to
`entropy_min_lambda` over `entropy_anneal_steps` learner calls.

Use this to encourage exploration early in training. Keep the coefficient small
enough that the policy can still become decisive when useful.

### ActorCriticConfig Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `device` | `"cpu"` | Device for the actor model and training tensors. |
| `regularizers` | `[]` | Optional callables that add extra losses. Each receives `(actor, rewards, dones, states, actions, next_states)`. |
| `reg_lams` | `[]` | Scalar weights for `regularizers`. Must have the same length as `regularizers`. |
| `num_actions` | `2` | Number of policy-logit columns read from the model output. |
| `clip_grad` | `True` | Enables gradient norm clipping after backpropagation. |
| `grad_norm_clip` | `1.0` | Maximum gradient norm used when `clip_grad=True`. |
| `use_bias` | `True` | Enables the value baseline and value loss. |
| `value_criterion` | `th.nn.MSELoss()` | Loss used for value estimates against value targets. |
| `value_lambda` | `0.1` | Weight applied to the value loss. |
| `value_targets` | `"td"` | Target type for the value head. Must be `"td"` or `"returns"`. |
| `gamma` | `0.99` | Discount used for bootstrapped advantages and TD value targets. |
| `advantage_bootstrap` | `True` | Uses one-step bootstrapped targets for advantages instead of full returns. |
| `use_entropy` | `False` | Enables entropy regularization. |
| `entropy_max_lambda` | `0.0` | Initial entropy coefficient. |
| `entropy_min_lambda` | `0.0` | Final entropy coefficient after annealing. |
| `entropy_anneal_steps` | `1` | Number of learner calls used for entropy annealing. Must be greater than `1` when entropy is enabled. |
| `normalize_advantages` | `False` | Normalizes advantages within each training batch. |

### PPOConfig Parameters

`PPOConfig` includes all `ActorCriticConfig` parameters plus:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `ppo_clipping` | `0.1` | Clipping range for the PPO ratio objective. |
| `ppo_iterations` | `4` | Number of optimization epochs over the collected rollout batch. |

Important validation rules:

- `regularizers` and `reg_lams` must have the same length.
- `advantage_bootstrap=True` requires `use_bias=True`.
- `value_targets` must be `"td"` or `"returns"`.
- When entropy is enabled, `entropy_anneal_steps` must be greater than `1`.
- When entropy is enabled, `entropy_max_lambda` must be greater than or equal to
  `entropy_min_lambda`.

### Actor-Critic Training Flow

`ActorCriticLearner.train(...)` performs one optimizer step:

1. Switches the actor to train mode.
2. Computes policy logits and values for the collected states.
3. Computes next-state values when bootstrapping or TD value targets need them.
4. Computes the policy loss from action probabilities and advantages.
5. Adds optional value loss, entropy loss, and regularization losses.
6. Clears gradients, backpropagates, optionally clips gradients, and steps the
   optimizer.
7. Advances the entropy annealing counter once per call.
8. Returns the scalar loss.

### OnPolicyExperiment Parameters

`OnPolicyExperiment` provides a full on-policy training loop.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_steps` | Required | Total environment transitions to collect before stopping. |
| `gamma` | `0.99` | Discount used by `Runner` when it computes returns. Keep aligned with `ActorCriticConfig.gamma` for return-based objectives. |
| `run_steps` | `0` | Number of transitions collected per runner call. `0` or less means collect one full episode. |
| `log_dir` | `"runs/on_policy_experiment"` | TensorBoard log directory. |
| `experiment_name` | `"OnPolicyExperiment"` | Progress bar label. |
| `step_callback` | `None` | Optional callable invoked at step `0` and then on configured step intervals. |
| `step_callback_interval` | `None` | Step spacing for `step_callback`. Required when a callback is provided. |

Important behavior:

- The on-policy experiment does not use a replay buffer.
- The learner trains immediately on each collected batch.
- The runner computes discounted returns only when the learner configuration
  needs them: return-based advantages, or return-based value targets with a
  value baseline.
- Positive `run_steps` can collect partial episodes. If full returns are needed
  and `run_steps` cuts an episode short, returns are computed over the collected
  segment rather than the unobserved remainder.

### Typical Actor-Critic Settings

For small discrete-control tasks, common starting points are:

- `use_bias=True`
- `advantage_bootstrap=True` for one-step actor-critic updates, or
  `advantage_bootstrap=False` with `run_steps=0` for full-episode returns.
- `value_targets="td"` for bootstrapped value learning, or `"returns"` when
  using complete episode returns.
- `normalize_advantages=True` when batch sizes are large enough to produce a
  meaningful mean and standard deviation.
- `use_entropy=True` with a small entropy coefficient when the policy collapses
  too early.
- Use `PPOLearner` when you want PPO-style repeated clipped-ratio updates.

## Controllers

Controllers convert model outputs into environment actions. Every controller
exposes:

```python
action = controller.choose(obs)
```

Discrete action controllers also expose:

```python
probs = controller.probabilities(obs)
```

### GreedyController

`GreedyController(model, num_actions)` selects the largest score among the
first `num_actions` model outputs:

```text
action = argmax_a output[a]
```

It is deterministic and is commonly used inside `EpsilonGreedyController` for
DQN.

Parameters:

| Parameter | Meaning |
| --- | --- |
| `model` | PyTorch model used to produce action scores. |
| `num_actions` | Number of leading output columns considered as valid actions. |

### EpsilonGreedyController

`EpsilonGreedyController` wraps another controller and sometimes replaces its
action with a random discrete action.

Epsilon is linearly annealed from `max_eps` to `min_eps`:

```text
frac = max(1 - num_decisions / (anneal_steps - 1), 0)
epsilon = frac * (max_eps - min_eps) + min_eps
```

Parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `controller` | Required | Base controller used for non-random decisions. |
| `num_actions` | Required | Number of uniformly sampled random actions. |
| `max_eps` | `1.0` | Initial random-action probability. |
| `min_eps` | `0.1` | Final random-action probability. |
| `anneal_steps` | `10_000` | Number of action decisions over which epsilon is annealed. Must be at least `2`. |

`choose(obs)` increments the internal decision counter by default. Pass
`increase_counter=False` when evaluating without advancing the exploration
schedule.

`probabilities(obs)` returns the mixture of the base controller's probabilities
and the uniform random policy. It does not increment the decision counter.

### StochasticController

`StochasticController(model, num_actions)` converts the first `num_actions`
model outputs into probabilities with `softmax` and samples from the resulting
categorical distribution.

Parameters:

| Parameter | Meaning |
| --- | --- |
| `model` | PyTorch model used to produce policy logits. |
| `num_actions` | Number of leading output columns used as policy logits. |

This is the natural controller for actor-critic training because it samples
actions from the learned policy.

### GaussianController

`GaussianController(model, action_dim)` reads means and log standard deviations
from the model output, samples a Gaussian action, and squashes it with `tanh`.
It is a `ContinuousActionController`, so it exposes `choose(obs)` but not
`probabilities(obs)`.

Parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model` | Required | PyTorch model used to produce Gaussian parameters. |
| `action_dim` | Required | Number of continuous action dimensions. |
| `deterministic` | `False` | Uses the mean action instead of sampling when `True`. |

## Runner

`Runner` collects transitions from a Gymnasium environment with a controller.

```python
batch, ep_returns, ep_lengths, last_episode = runner.run(num_steps)
```

Constructor parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `env` | Required | Gymnasium environment. |
| `controller` | Required | Controller used to choose actions. |
| `calculate_returns` | `True` | Computes discounted returns for collected transitions. |
| `return_last_episode` | `True` | Returns the latest complete episode as a separate `TransitionBatch`. |
| `gamma` | `0.99` | Discount used for return calculation. |
| `device` | `"cpu"` | Device used for observations passed to the controller. |
| `continuous_actions` | `False` | Converts controller outputs to clipped NumPy action vectors for continuous environments. |

`run(num_steps)` behavior:

- `num_steps <= 0` collects one complete episode.
- `num_steps > 0` collects up to that many transitions and may return a partial
  episode.
- `ep_returns` and `ep_lengths` contain only episodes completed during the run.
- `last_episode` is `None` unless `return_last_episode=True` and an episode
  ended during the run.

## Replay Buffer

`ReplayBuffer` stores transitions in fixed-size NumPy arrays and returns
`TransitionBatch` objects on request.

Constructor parameters:

| Parameter | Meaning |
| --- | --- |
| `capacity` | Maximum number of transitions retained. New data overwrites old data in a circular buffer. |
| `obs_shape` | Shape of one observation, excluding the batch dimension. |
| `device` | Device used for sampled tensors. |

Methods:

| Method | Meaning |
| --- | --- |
| `add(...)` | Adds a batch of transitions. The added batch length must be no larger than `capacity`. |
| `get_all()` | Returns all currently stored transitions. |
| `sample(batch_size)` | Uniformly samples random indices from the stored transitions and returns a `TransitionBatch`. |

The buffer stores actions as `int64`, rewards and returns as `float32`, and
dones as boolean values.

## Choosing Between Algorithms

Use DQN when:

- The action space is discrete.
- You want off-policy learning from a replay buffer.
- Deterministic greedy evaluation is appropriate.
- You can represent the policy as action values.

Use actor-critic when:

- The action space is discrete and a stochastic policy is useful.
- You want on-policy policy-gradient updates.
- You want a value baseline, entropy regularization, or, with `PPOLearner`,
  PPO-style repeated updates.
- You prefer directly optimizing action probabilities instead of Q-values.

The algorithms in this package assume discrete action spaces. Continuous
action algorithms are not currently implemented.
