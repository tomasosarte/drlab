# Examples

Run these examples from the repository root after installing `drlab`.

## CartPole

Short examples using the high-level experiment wrappers:

```bash
python examples/dqn_cartpole.py
python examples/reinforce_cartpole.py
python examples/actor_critic_cartpole.py
python examples/ppo_cartpole.py
```

Compare drlab and Stable-Baselines3 on the same workload:

```bash
python -m pip install stable-baselines3
python examples/compare_drlab_stable_baselines3.py --steps 10000
```

## MetaWorld

Train SAC on one MetaWorld task with a Continual-World-style configuration:

```bash
python -m pip install metaworld
python examples/sac_metaworld.py --task reach --device cuda
```

The default run uses one million environment steps. Use `--steps` for a shorter
run. The optimizer defaults to fused Adam on CUDA and regular Adam on CPU.

## TensorBoard

TensorBoard logs are written under `runs/examples/`.

```bash
tensorboard --logdir runs/examples
```
