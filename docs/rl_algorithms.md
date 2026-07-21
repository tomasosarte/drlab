# RL Algorithms in drlab

This document explains the reinforcement learning algorithms implemented in
`drlab`, how they map to the library components, and which environments they
are designed to support.

`drlab` is not trying to hide the algorithm behind a large framework. The usual
workflow is explicit: define a PyTorch model, wrap it with a controller, collect
data from a Gymnasium environment, and train a learner through an experiment
loop.

## Shared Concepts

The main pieces used by all algorithms are:

- `Runner`: steps through a Gymnasium environment with a controller and returns
  transition data.
- `TransitionBatch`: stores tensors for states, actions, rewards, termination and
  truncation flags, next states, and returns.
- `Controller`: converts model outputs into environment actions.
- `Learner`: updates one or more PyTorch models from transition batches.
- `Experiment`: coordinates the runner, learner, logging, and replay-buffer
  behavior when needed.

There are two training families:

- **On-policy** algorithms train from data collected by the current policy. In
  `drlab`, these use `OnPolicyExperiment`.
- **Off-policy** algorithms can train from older transitions stored in a replay
  buffer. In `drlab`, these use `OffPolicyExperiment`.

The library supports both discrete and continuous action spaces. Discrete
actions are stored as integer action indices. Continuous actions are stored as
floating point action vectors.

## Supported Algorithms

| Algorithm | Learner | Type | Action Space | Replay Buffer | Typical Controller |
| --- | --- | --- | --- | --- | --- |
| DQN | `DQNLearner` | Off-policy value learning | Discrete | Yes | `EpsilonGreedyController` |
| SAC | `SACLearner` | Off-policy actor-critic | Continuous | Yes | `GaussianController` |
| REINFORCE | `ReinforceLearner` | On-policy policy gradient | Discrete | No | `StochasticController` |
| Actor-Critic | `ActorCriticLearner` | On-policy actor-critic | Discrete | No | `StochasticController` |
| PPO | `PPOLearner` | On-policy clipped policy gradient | Discrete | No | `StochasticController` |

## Data and Model Conventions

Learners operate on batches with shape `[B, ...]`, where `B` is the batch size.

| Field | Shape | Meaning |
| --- | --- | --- |
| `states` | `[B, *obs_shape]` | Current observations. |
| `actions` | `[B, 1]` or `[B, *action_shape]` | Discrete action indices or continuous action vectors. |
| `rewards` | `[B, 1]` | Immediate rewards. |
| `terminated` | `[B, 1]` | True terminal-state flags; these disable value bootstrapping. |
| `truncated` | `[B, 1]` | External episode cutoffs, such as time limits; these do not disable bootstrapping. |
| `next_states` | `[B, *obs_shape]` | Observations after each action. |
| `returns` | `[B, 1]` | Discounted returns, used by on-policy methods when needed. |

Model output conventions depend on the learner or controller:

| Component | Expected model output |
| --- | --- |
| DQN and greedy controllers | At least `num_actions` columns. The first columns are Q-values or action scores. |
| REINFORCE and stochastic controllers | At least `num_actions` columns. The first columns are policy logits. |
| Actor-critic and PPO | At least `num_actions + 1` columns. The first columns are policy logits, followed by one value estimate. |
| SAC actor and Gaussian controller | `2 * action_dim` columns: means followed by log standard deviations. |
| SAC critics | A scalar Q-value from concatenated state-action input. |

## Off-Policy Algorithms

Off-policy learners are trained from replay-buffer samples. The environment may
be generating new behavior with an exploratory controller while the learner
updates from transitions collected earlier.

`OffPolicyExperiment` handles this loop:

1. Use `Runner` to collect transitions.
2. Add the transitions to a `ReplayBuffer`.
3. Wait until the buffer has enough samples for a batch.
4. Sample minibatches from the buffer.
5. Call the learner one or more times.

`OffPolicyExperimentConfig` accepts these parameters:

| Parameter | Description |
| --- | --- |
| `max_steps` | Total number of environment steps to collect. |
| `gamma` | Discount factor passed to the runner. Defaults to `0.99`. |
| `run_steps` | Steps collected per runner call; values `<= 0` run one episode. Defaults to `0`. |
| `log_dir` | TensorBoard output directory. |
| `experiment_name` | Name displayed by the progress bar. |
| `use_replay` | Whether to sample training batches from replay. Defaults to `True`. |
| `replay_buffer_size` | Maximum number of stored transitions. Defaults to `10_000`. |
| `batch_size` | Number of transitions in a training batch. Defaults to `128`. |
| `use_last_episode` | Whether to include the latest episode in the training batch. Defaults to `True`. |
| `grad_repeats` | Learner updates performed per collected batch. Defaults to `1`. |
| `step_callback` | Optional callback receiving the current step. |
| `step_callback_interval` | Number of steps between callback calls. |
| `learning_starts` | Step at which learning begins. Defaults to `batch_size`. |
| `warmup_steps` | Initial steps that use random actions sampled from the environment. Defaults to `0`. |

### DQN

Deep Q-Network learns a value for each discrete action. Given a state, the model
outputs Q-values, and the chosen action is usually the one with the highest
value. During training, DQN compares the current Q-value against a one-step
temporal-difference target:

```text
target = reward + gamma * (1 - terminated) * next_q
loss = criterion(Q(state, action), target)
```

In `drlab`, `DQNLearner` is usually paired with:

- `EpsilonGreedyController` for exploration.
- `GreedyController` as the base action selector.
- `OffPolicyExperiment` for replay-buffer training.

Important implementation details:

- The online model predicts current Q-values.
- A target model can be used to compute next-state values more stably.
- Target models can be updated softly every training step or copied with hard
  updates every fixed number of learner calls.
- Double DQN is supported. It selects the next action with the online model and
  evaluates that action with the target model.
- DQN is for discrete action spaces.

Useful config fields include:

- `num_actions`: number of discrete actions read from the model output.
- `gamma`: discount factor for the TD target.
- `criterion`: loss used between predicted Q-values and TD targets.
- `double_q`: enables Double DQN action selection.
- `use_target_model`: enables the target Q-network.
- `target_update`: `"soft"` or `"hard"`.
- `soft_target_update_param`: soft-update coefficient.
- `target_update_interval`: hard-update interval.
- `clipnorm`: maximum gradient norm, or `None` to disable clipping.

### SAC

Soft Actor-Critic is an off-policy actor-critic algorithm for continuous action
spaces. It trains:

- an actor that samples continuous actions,
- two critics that estimate Q-values for state-action pairs,
- two target critics for stable bootstrapping,
- and an entropy coefficient that controls exploration pressure.

The actor outputs Gaussian parameters. Actions are sampled with the
reparameterization trick and squashed with `tanh`, so they are bounded in
`[-1, 1]`.

The critic target combines reward, future value, and entropy:

```text
target_q = min(Q1_target(next_state, next_action),
               Q2_target(next_state, next_action))

target = reward + gamma * (1 - terminated) * (target_q - alpha * log_prob)
```

The actor is trained to prefer actions with high predicted value while also
keeping enough entropy for exploration:

```text
actor_loss = mean(alpha * log_prob - min(Q1, Q2))
```

In `drlab`, `SACLearner` is usually paired with:

- `GaussianController` for continuous action sampling.
- `OffPolicyExperiment` for replay-buffer training.
- A Gymnasium environment with a continuous `Box` action space.

Important config fields include:

- `action_shape`: continuous action shape.
- `gamma`: discount factor.
- `criterion`: critic loss.
- `target_entropy`: desired policy entropy. If unset, it defaults to
  `-action_dim`.
- `initial_alpha`: initial entropy coefficient before automatic tuning. It
  defaults to `1.0`.
- `alpha_lr`: learning rate for the entropy coefficient.
- `min_log_std` and `max_log_std`: clamp bounds for actor log standard
  deviations.
- `soft_target_update_param`: soft-update coefficient for target critics.

## On-Policy Algorithms

On-policy learners train on fresh batches collected from the current policy.
They do not use replay-buffer sampling. `OnPolicyExperiment` collects rollout
data, computes returns when required, and calls the learner update.

### REINFORCE

REINFORCE is the simplest policy-gradient algorithm in `drlab`. The model
outputs policy logits for discrete actions. The controller samples actions from
the policy, and the learner increases the probability of actions that led to
higher returns.

The core loss is:

```text
loss = -mean(log pi(action | state) * return)
```

In `drlab`, `ReinforceLearner` is usually paired with:

- `StochasticController`.
- `OnPolicyExperiment`.
- A model that outputs one policy-logit column per discrete action.

REINFORCE is useful as a simple baseline. It is easy to understand, but it can
have high variance because it uses returns directly instead of a learned value
baseline.

Useful config fields include:

- `num_actions`: number of discrete action logits.
- `normalize_returns`: normalizes returns inside the batch before computing the
  policy-gradient loss.
- `use_entropy`: adds entropy regularization for exploration.
- `clip_grad` and `clipnorm`: gradient clipping settings.

### Actor-Critic

Actor-critic adds a value estimate to the policy model. The same network emits:

```text
output[:, :num_actions]              -> policy logits
output[:, num_actions:num_actions+1] -> value estimate
```

The policy is trained with an advantage instead of the raw return:

```text
advantage = target_value - V(state)
policy_loss = -mean(log pi(action | state) * advantage)
```

When the value baseline is enabled, the value head is also trained:

```text
value_loss = criterion(V(state), target_value)
```

In `drlab`, `ActorCriticLearner` is usually paired with:

- `StochasticController`.
- `OnPolicyExperiment`.
- A model with policy logits and one value column.

Important implementation choices:

- Advantages can come from one-step TD targets or full discounted returns.
- The value head can be trained against TD targets or returns.
- Advantage normalization is available.
- Entropy regularization can be enabled and annealed over learner calls.

Useful config fields include:

- `use_bias`: enables the learned value baseline and value loss.
- `value_targets`: `"td"` or `"returns"`.
- `value_lambda`: weight of the value loss.
- `gamma`: discount factor for TD targets.
- `advantage_bootstrap`: uses one-step bootstrapped advantages instead of full
  returns.
- `normalize_advantages`: normalizes advantages inside the batch.

### PPO

Proximal Policy Optimization builds on actor-critic. It still uses policy
logits, value estimates, and advantages, but it reuses the same rollout batch
for several optimization epochs. To avoid changing the policy too aggressively,
it compares the new action probability with the old one and clips the update.

The main policy objective uses:

```text
ratio = pi_new(action | state) / pi_old(action | state)
clipped_ratio = clamp(ratio, 1 - clip, 1 + clip)
policy_loss = -mean(min(ratio * advantage,
                        clipped_ratio * advantage))
```

In `drlab`, `PPOLearner` is usually paired with:

- `StochasticController`.
- `OnPolicyExperiment`.
- The same model output layout as `ActorCriticLearner`.

This implementation is intentionally compact. It stores old action
probabilities before the first update, then runs `ppo_iterations` optimization
passes over the same batch. It does not currently add generalized advantage
estimation, minibatch shuffling, or a separate old-policy network.

Useful config fields include:

- `ppo_clipping`: clipping range around the probability ratio.
- `ppo_iterations`: number of optimization passes over each rollout batch.
- all relevant `ActorCriticConfig` fields.

## Entropy and Regularization

On-policy learners can use entropy regularization. Entropy encourages broader
action distributions and is often useful early in training:

```text
entropy_loss = -entropy_lambda * policy_entropy
```

`entropy_lambda` is linearly annealed from `entropy_max_lambda` to
`entropy_min_lambda` over `entropy_anneal_steps`.

Both on-policy and off-policy base learners also support custom regularizers.
A regularizer receives the model and transition tensors, returns a scalar loss,
and is weighted by the matching value in `reg_lams`.

## Choosing an Algorithm

- Use DQN for simple discrete-action environments and value-based baselines.
- Use SAC for continuous-control environments with bounded action spaces.
- Use REINFORCE when you want the simplest policy-gradient baseline.
- Use actor-critic when you want an on-policy method with a learned value
  baseline.
- Use PPO when you want a more stable on-policy policy-gradient method that can
  reuse rollout data for multiple updates.

## Typical Pairings

| Goal | Suggested Components |
| --- | --- |
| Discrete off-policy control | `DQNLearner` + `EpsilonGreedyController` + `OffPolicyExperiment` |
| Continuous off-policy control | `SACLearner` + `GaussianController` + `OffPolicyExperiment` |
| Simple policy-gradient baseline | `ReinforceLearner` + `StochasticController` + `OnPolicyExperiment` |
| On-policy value-baseline method | `ActorCriticLearner` + `StochasticController` + `OnPolicyExperiment` |
| Clipped on-policy optimization | `PPOLearner` + `StochasticController` + `OnPolicyExperiment` |

For runnable examples, see the `examples/` directory.
