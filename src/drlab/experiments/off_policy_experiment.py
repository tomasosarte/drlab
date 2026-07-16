import numpy as np
from tqdm import tqdm
import gymnasium as gym
from dataclasses import dataclass
from typing import Callable
from torch.utils.tensorboard import SummaryWriter

from drlab.learners import OffPolicyLearner, SACLearner
from drlab.runners import Runner
from drlab.controllers import Controller, WarmupController
from drlab.replay import TransitionBatch, ReplayBuffer

@dataclass
class OffPolicyExperimentConfig:
    max_steps: int
    gamma: float = 0.99
    run_steps: int = 0
    log_dir: str = "runs/off_policy_experiment"
    experiment_name: str = "OffPolicyExperiment"
    use_replay: bool = True
    replay_buffer_size: int = 10_000
    batch_size: int = 128
    use_last_episode: bool = True
    grad_repeats: int = 1
    step_callback: Callable[[int], None] | None = None
    step_callback_interval: int | None = None
    learning_starts: int | None = None
    warmup_steps: int = 0


class OffPolicyExperiment:

    def __init__(
            self,
            env: gym.Env,
            controller: Controller,
            learner: OffPolicyLearner,
            config: OffPolicyExperimentConfig,
        ):

        # Init experiment settings
        self.max_steps = config.max_steps
        self.grad_repeats = config.grad_repeats
        self.batch_size = config.batch_size
        self.use_last_episode = config.use_last_episode
        self.run_steps = config.run_steps
        self.step_callback = config.step_callback
        self.step_callback_interval = config.step_callback_interval
        self.steps = 0
        self.learning_starts = (
            config.batch_size if config.learning_starts is None else config.learning_starts
        )
        self.warmup_steps = config.warmup_steps

        self._validate_config()

        # Init drl components
        continous_actions = isinstance(learner, SACLearner)
        if self.warmup_steps > 0:
            controller = WarmupController(
                controller,
                env.action_space,
                warmup_steps=self.warmup_steps,
            )
        self.runner = Runner(
            env,
            controller,
            False,
            config.use_last_episode,
            config.gamma,
            learner.device,
            continous_actions,
        )
        self.learner = learner

        # Init replay buffer
        self.replay_buffer_size = config.replay_buffer_size
        self.use_replay = config.use_replay
        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_size if config.use_replay else config.batch_size,  
            obs_shape=env.observation_space.shape,
            device=learner.device,
            continuous_actions=continous_actions,
            action_shape=learner.config.action_shape
        )

        # Init logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.experiment_name = config.experiment_name


    def _validate_config(self):
        if self.step_callback is not None and self.step_callback_interval is None:
            raise ValueError("step_callback_interval must be set when step_callback is provided.")

        if self.learning_starts < self.batch_size:
            raise ValueError("learning_starts must be greater than or equal to batch_size.")

        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be greater than or equal to 0.")

    def _make_minibatch(self, batch: TransitionBatch, last_episode: TransitionBatch | None) -> TransitionBatch:

        # No replay buffer
        if not self.use_replay:
            return self.replay_buffer.get_all()
        
        # Replay buffer without last episode
        if not self.use_last_episode:
            return self.replay_buffer.sample(self.batch_size)

        # Replay buffer with last episode
        episode = last_episode if last_episode else batch
        episode = episode.to(self.learner.device)

        ep_len = episode.states.shape[0]
        if ep_len >= self.batch_size:
            return episode

        rest = self.replay_buffer.sample(self.batch_size - ep_len)
        return rest.cat(episode)

    def _learn_from_batch(
        self,
        batch: TransitionBatch,
        last_episode: TransitionBatch | None,
    ) -> float:
        # 1) Add batch to replay buffer
        self.replay_buffer.add(
            states=batch.states,
            actions=batch.actions,
            rewards=batch.rewards,
            terminated=batch.terminated,
            truncated=batch.truncated,
            next_states=batch.next_states,
            returns=batch.returns,
        )
        if self.steps < self.learning_starts:
            return 0.0

        # 2) train repeats
        total_loss = 0.0
        for _ in range(self.grad_repeats):
            mb = self._make_minibatch(batch, last_episode)
            total_loss += self.learner.train(
                mb.rewards, mb.terminated, mb.states, mb.actions, mb.next_states
            )

        return total_loss / self.grad_repeats

    def run(self):
        
        self.steps = 0
        if self.step_callback is not None:
            self.step_callback(self.steps)
        next_callback_step = self.step_callback_interval
        pbar = tqdm(total=self.max_steps, desc=self.experiment_name, dynamic_ncols=True, mininterval=1.0)
        while self.steps < self.max_steps:

            # 1. Run batch of transitions
            batch, ep_returns, ep_lengths, last_episode = self.runner.run(
                self.run_steps,
                as_numpy=True,
            )

            step_inc = batch.states.shape[0]
            self.steps += step_inc
            pbar.update(step_inc)

            # 2. Learn from batch
            loss = self._learn_from_batch(batch, last_episode)

            # 3. Log results
            self.writer.add_scalar("Loss", loss, self.steps)
            if ep_returns:
                mean_return = np.mean(ep_returns)
                mean_length = np.mean(ep_lengths)
                self.writer.add_scalar("Episode return", mean_return, self.steps)
                self.writer.add_scalar("Episode Length", mean_length, self.steps)

                pbar.set_postfix({
                    "loss": f"{loss:.3f}",
                    "return": f"{mean_return:.2f}",
                    "len": f"{mean_length:.1f}",
                }, refresh=False)

            while self.step_callback is not None and self.steps >= next_callback_step:
                self.step_callback(next_callback_step)
                next_callback_step += self.step_callback_interval
        
        pbar.close()
        self.writer.close()
