# drlab examples

These examples train each learner on `CartPole-v1`, a small built-in Gymnasium
environment. They are intentionally short and use the high-level experiment
wrappers.

Run them from the repository root after installing `drlab`:

```bash
python examples/dqn_cartpole.py
python examples/reinforce_cartpole.py
python examples/actor_critic_cartpole.py
python examples/ppo_cartpole.py
```

There is also a comparison script that trains drlab DQN and Stable-Baselines3
DQN for the same number of `CartPole-v1` steps and evaluates both policies:

```bash
python -m pip install stable-baselines3
python examples/compare_drlab_stable_baselines3.py --steps 10000
```

TensorBoard logs are written under `runs/examples/`.

```bash
tensorboard --logdir runs/examples
```

The step counts are small enough for a quick demo, not tuned benchmarks. Increase
`max_steps` in a script if you want longer training.
