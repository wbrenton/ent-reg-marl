# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An example of use of PPO.

Note: code adapted (with permission) from
https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py and
https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_atari.py
"""

# pylint: disable=g-importing-member
import collections
from datetime import datetime
import logging
import os
import random
import sys
import time
from pathlib import Path
from rich.pretty import pprint
from absl import app
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import pyspiel
from open_spiel.python.rl_environment import ChanceEventSampler
from open_spiel.python.rl_environment import Environment
from open_spiel.python.rl_environment import ObservationType
from open_spiel.python.vector_env import SyncVectorEnv
from ppo import PPO, PPOAgent

import dataclasses
import os

@dataclasses.dataclass
class Args:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    game_name: str = "dark_hex"
    learning_rate: float = 2.5e-4
    seed: int = 1
    total_timesteps: int = 10_000_000
    eval_every: int = 100
    torch_deterministic: bool = True
    cuda: bool = True

    # Algorithm specific arguments
    num_envs: int = 1
    num_steps: int = 128
    anneal_lr: bool = False
    gae: bool = True
    gamma: float = 1.0 # 0.99
    gae_lambda: float = 1.0 # 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = None


def make_single_env(game_name, seed):

  def gen_env():
    game = pyspiel.load_game(game_name)
    return Environment(game, chance_event_sampler=ChanceEventSampler(seed=seed))

  return gen_env


def main(_):
  args = Args()
  pprint(dataclasses.asdict(args))
  exp_name = f"ppo_flip_lr_{args.learning_rate}_norm_adv_{args.norm_adv}_gamma_{args.gamma}_ent_coef_{args.ent_coef}/"

  batch_size = int(args.num_envs * args.num_steps)
  current_day = datetime.now().strftime("%d")
  current_month_text = datetime.now().strftime("%h")
  run_name = f"{args.game_name}__{args.exp_name}__"
  run_name += f"{args.seed}__{current_month_text}__{current_day}__{int(time.time())}"

  writer = SummaryWriter(f"runs/{run_name}")
  writer.add_text(
      "hyperparameters",
      "|param|value|\n|-|-|\n%s" %
      ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
  )

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = args.torch_deterministic

  device = torch.device(
      "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
  print("Using device: %s", str(device))

  envs = SyncVectorEnv([
      make_single_env(args.game_name, args.seed + i)()
      for i in range(args.num_envs)
  ])
  agent_fn = PPOAgent

  game = envs.envs[0]._game  # pylint: disable=protected-access
  info_state_shape = envs.observation_spec()["info_state"][0]
  # info_state_shape = game.observation_tensor_shape()

  num_updates = args.total_timesteps // batch_size
  agent = PPO(
      input_shape=info_state_shape,
      num_actions=game.num_distinct_actions(),
      num_players=game.num_players(),
      num_envs=args.num_envs,
      steps_per_batch=args.num_steps,
      num_minibatches=args.num_minibatches,
      update_epochs=args.update_epochs,
      learning_rate=args.learning_rate,
      gae=args.gae,
      gamma=args.gamma,
      gae_lambda=args.gae_lambda,
      normalize_advantages=args.norm_adv,
      clip_coef=args.clip_coef,
      clip_vloss=args.clip_vloss,
      entropy_coef=args.ent_coef,
      value_coef=args.vf_coef,
      max_grad_norm=args.max_grad_norm,
      target_kl=args.target_kl,
      device=device,
      writer=writer,
      agent_fn=agent_fn,
  )

  n_reward_window = 1000 # last 1000 games
  rewards = []
  time_step = envs.reset()
  for update in range(num_updates):
    for _ in range(args.num_steps):
      pid = [ts.current_player() for ts in time_step]
      agent_output = agent.step(time_step)
      time_step, reward, done, unreset_time_steps = envs.step(
          agent_output, reset_if_done=True)

      for ts, actor_pid in zip(unreset_time_steps, pid):
        reward = ts.rewards[actor_pid]
        done = ts.last()
        if done:
          writer.add_scalar("charts/player_0_training_returns", ts.rewards[0],
                            agent.total_steps_done)
          rewards.append(ts.rewards[0])

      agent.post_step(reward, done)

    if args.anneal_lr:
      agent.anneal_learning_rate(update, num_updates)

    agent.learn(time_step)

    if update % args.eval_every == 0:
      print("-" * 80)
      print(f"Step: {agent.total_steps_done} mean reward: {np.array(rewards).mean() if len(rewards) < n_reward_window else np.array(rewards[-n_reward_window:]).mean()} all time {np.array(rewards).mean()}", flush=True)

  here = os.path.dirname(os.path.abspath(__file__)) 
  directory = f"{here}/../../results/{exp_name}/{args.game_name}/"
  Path(directory).mkdir(parents=True, exist_ok=True)
  agent.save(directory)
  
  rewards = np.array(rewards)
  # save as npy
  np.save(directory + f"rewards_{args.seed}.npy", rewards)

  writer.close()
  print("All done. Have a pleasant day :)")


if __name__ == "__main__":
  app.run(main)