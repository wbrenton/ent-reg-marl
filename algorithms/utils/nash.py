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
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability


class EvaluateAveragePolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, policies, mode):
    game = env.game
    player_ids = [0, 1]
    super(EvaluateAveragePolicies, self).__init__(game, player_ids)
    self._policies = policies
    self._mode = mode
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    with self._policies[cur_player].temp_mode_as(self._mode):
      p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


class EvaluationPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, policies):
    game = env.game
    player_ids = [0, 1]
    super(EvaluationPolicies, self).__init__(game, player_ids)
    if len(policies) != 2:
        import copy 
        policies = [copy.deepcopy(policies[0]), copy.deepcopy(policies[0])]
    self._policies = policies
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def train_and_evaluate_nash(args, env, agents):
    # pylint: disable=g-complex-comprehension
    if args.exp_name in ["a2c", "ppo"]:
        expl_policies_avg = EvaluationPolicies(env, agents)
    elif args.exp_name == "nfsp":
        expl_policies_avg = EvaluateAveragePolicies(env, agents, "average_policy")
    else:
        raise ValueError("Unknown experiment name: {}".format(args.exp_name))

    here = os.path.dirname(os.path.abspath(__file__))
    directory = f"{here}/../../results/{args.exp_name}/{args.game}/"
    Path(directory).mkdir(parents=True, exist_ok=True)
    fn = directory + f"nash_10m"
    expls = []
    env_steps = 0
    while env_steps < args.num_env_steps:
        time_step = env.reset()
        while not time_step.last():
            if (env_steps + 1) % args.eval_every == 0:
                losses = [agent.loss for agent in agents]
                expl = exploitability.exploitability(env.game, expl_policies_avg)
                expls.append([(env_steps + 1), expl.item()])
                print((env_steps + 1), expl.item())
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)
            env_steps += 1
        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)

    for agent in agents:
        agent.save(fn)

    expls = np.array(expls)
    df = pd.DataFrame(data={'Exploitability': expls[:, 1]}, index=expls[:, 0].astype(np.int32))
    df.to_csv(fn + '.csv')
    df.plot()
    plt.xlabel("Ep")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(fn + ".png")