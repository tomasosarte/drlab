import numpy as np
from tqdm import tqdm
import gymnasium as gym
from dataclasses import dataclass
from typing import Callable
from torch.utils.tensorboard import SummaryWriter

from drlab.learners import OnPolicyLearner
from drlab.runners import Runner
from drlab.controllers import Controller


@dataclass
class OnPolicyExperimentConfig:
    max_steps: int
    gamma: float = 0.99
    run_steps: int = 0
    log_dir: str = "runs/on_policy_experiment"
    experiment_name: str = "OnPolicyExperiment"
    step_callback: Callable[[int], None] | None = None
    step_callback_interval: int | None = None


class OnPolicyExperiment:

    def __init__(
        self,
        env: gym.Env,
        controller: Controller,
        learner: OnPolicyLearner,
        config: OnPolicyExperimentConfig,
    ):  
        # Init experiment settings
        self.max_steps = config.max_steps
        self.run_steps = config.run_steps
        self.step_callback = config.step_callback
        self.step_callback_interval = config.step_callback_interval
        
        calculate_returns = learner.requires_returns()

        # Init drl components
        self.runner = Runner(env, controller, calculate_returns, False, config.gamma, learner.device)
        self.learner = learner

        # Init logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.experiment_name = config.experiment_name

        if self.step_callback is not None and self.step_callback_interval is None:
            raise ValueError("step_callback_interval must be set when step_callback is provided.")

    def run(self):

        steps = 0
        if self.step_callback is not None:
            self.step_callback(steps)
        next_callback_step = self.step_callback_interval
        pbar = tqdm(total=self.max_steps, desc=self.experiment_name, dynamic_ncols=True, mininterval=1.0)
        while steps < self.max_steps:
            
            # 1. Run batch of transitions
            batch, ep_returns, ep_lengths, _ = self.runner.run(self.run_steps)
            batch = batch.to(self.learner.device)

            # 2. Learn from batch
            loss = self.learner.train(
                batch.rewards,
                batch.dones,
                batch.states,
                batch.actions,
                batch.next_states,
                batch.returns,
            )

            # 3. Log results
            self.writer.add_scalar("Loss", loss, steps)
            if ep_returns:
                mean_return = np.mean(ep_returns)
                mean_length = np.mean(ep_lengths)
                self.writer.add_scalar("Episode return", mean_return, steps)
                self.writer.add_scalar("Episode Length", mean_length, steps)

                pbar.set_postfix({
                    "loss": f"{loss:.3f}",
                    "return": f"{mean_return:.2f}",
                    "len": f"{mean_length:.1f}",
                }, refresh=False)

            step_inc = batch.states.shape[0]
            steps += step_inc
            pbar.update(step_inc)

            while self.step_callback is not None and steps >= next_callback_step:
                self.step_callback(next_callback_step)
                next_callback_step += self.step_callback_interval

        pbar.close()
        self.writer.close()
