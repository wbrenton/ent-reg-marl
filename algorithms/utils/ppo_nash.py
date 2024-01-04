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
import random
import torch

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability


class EvaluationPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, policy):
    game = env.game
    player_ids = [0, 1]
    super(EvaluationPolicies, self).__init__(game, player_ids)
    self._policies = [policy, policy]
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


def train_and_evaluate_nash(args, env, agent):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # pylint: disable=g-complex-comprehension
    expl_policies_avg = EvaluationPolicies(env, agent)

    here = os.path.dirname(os.path.abspath(__file__))
    directory = f"{here}/../../results/{args.exp_name}/{args.game}/"
    Path(directory).mkdir(parents=True, exist_ok=True)
    fn = directory + f"nash_10m"
    expls = []
    env_steps = 0
    num_updates = args.num_env_steps // args.num_steps
    time_step = env.reset()
    for update in range(num_updates):
        for step in range(args.num_steps):
            if (env_steps + 1) % args.eval_every == 0:
                losses = agent.loss
                expl = exploitability.exploitability(env.game, expl_policies_avg)
                expls.append([(env_steps + 1), expl.item()])
                print(f"Update: {update} Env Step: {(env_steps + 1)}, Expl: {expl.item()} Loss: {losses}")
            pid = time_step.current_player()
            agent_output = agent.step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)
            agent.post_step(time_step.rewards[pid], 1 - time_step.discounts[pid])
            env_steps += 1
            if time_step.last():
                time_step = env.reset()
        if args.anneal_lr:
            agent.anneal_learning_rate(update, num_updates)
        agent.learn(time_step)
        # if len(expls) > 20:
        #     expls_arr = np.array(expls)
        #     if expls_arr[-20:-10, 1].mean() - expls_arr[-10:-1, 1].mean() < 0.01:
        #         print(f"Trained for {env_steps} steps")
        #         break
    agent.save(fn)

    expls = np.array(expls)
    df = pd.DataFrame(data={'Exploitability': expls[:, 1]}, index=expls[:, 0].astype(np.int32))
    df.to_csv(fn + '.csv')
    df.plot()
    plt.xlabel("Ep")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(fn + ".png")

    return expls[-10:-1, 1].mean()