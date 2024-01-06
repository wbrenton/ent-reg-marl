# Copyright 2019 DeepMind Technologies Limited
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

"""
Train and evaluated on Kuhn Poker. 
Adapted from openspiel.pythton.examples.kuhn_policy_gradient
"""

import os
import copy
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from open_spiel.python.algorithms import random_agent

# from open_spiel.python.examples.rl_response
def eval_against_fixed_bots(env, trained_agents, fixed_agents, num_episodes=1_000):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(fixed_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = fixed_agents[:]
    cur_agents[player_pos] = trained_agents[player_pos]
    for _ in range(num_episodes):
      time_step = env.reset()
      episode_rewards = 0
      turn_num = 0
      while not time_step.last():
        turn_num += 1
        player_id = time_step.observations["current_player"]
        if env.is_turn_based:
          agent_output = cur_agents[player_id].step(
              time_step, is_evaluation=True)
          action_list = [agent_output.action]
        else:
          agents_output = [
              agent.step(time_step, is_evaluation=True) for agent in cur_agents
          ]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  return sum_episode_rewards / num_episodes


def train_and_evaluate_against_random(args, env, agent):
    fixed_agents = [
        random_agent.RandomAgent(
            player_id=idx,
            num_actions=env.action_spec()["num_actions"])
        for idx in range(args.num_players)
    ]

    eval_env = copy.deepcopy(env)
    here = os.path.dirname(os.path.abspath(__file__))
    directory = f"{here}/../../results/{args.exp_name}/{args.game}/"
    Path(directory).mkdir(parents=True, exist_ok=True)
    fn = directory + f"_vs_random"
    returns = []
    env_steps = 0
    num_updates = args.num_env_steps // args.steps_per_batch
    time_step = env.reset()
    for update in range(num_updates):
      for step in range(args.steps_per_batch):
        if (env_steps + 1) % args.eval_every == 0:
          r_p0, r_p1 = eval_against_fixed_bots(eval_env, [agent, agent], fixed_agents)
          returns.append([(env_steps + 1), r_p0, r_p1])
          agent.writer.add_scalar("charts/reward_pid0", r_p0, env_steps + 1)
          agent.writer.add_scalar("charts/reward_pid1", r_p1, env_steps + 1)
          agent.writer.add_scalar("charts/reward_mean", (r_p0 + r_p1) / 2, env_steps + 1)
          print(f"Update: {update} Env Step: {env_steps + 1}, Losses: {agent.loss}, Returns: {r_p0}, {r_p1}", flush=True)
        player_id = time_step.observations["current_player"]
        agent_output = agent.step(time_step)
        action_list = [agent_output.action]
        time_step = env.step(action_list)
        agent.post_step(time_step.rewards[player_id], time_step.last())
        env_steps += 1
        if time_step.last():
          time_step = env.reset()
      agent.learn(time_step)
    agent.save(directory)
    returns = np.array(returns)
    df = pd.DataFrame(data={'pid_0': returns[:, 1], 'pid_1': returns[:, 2]}, index=returns[:, 0].astype(np.int32))
    df['mean'] = df.mean(axis=1)
    df.to_csv(fn + '.csv')
    df.plot()
    plt.xlabel("Ep")
    plt.savefig(fn + "_linear.png")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(fn + "_log.png")