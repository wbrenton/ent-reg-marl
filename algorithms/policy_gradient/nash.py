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
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from open_spiel.python.jax import policy_gradient

game_choices = [
    "kuhn_poker",
    "dark_hex(board_size=2,gameversion=adh)",
    "liars_dice(dice_sides=4)",
    "leduc_poker",
]

loss_choices = ["a2c", "rpg", "qpg", "rm"]


class PolicyGradientPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies):
    game = env.game
    player_ids = [0, 1]
    super(PolicyGradientPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game",
        default="kuhn_poker",
        choices=game_choices,
    )
    parser.add_argument("--num_episodes", default=1_000_0)
    parser.add_argument("--eval_every", default=1_000)
    parser.add_argument("--loss_str", default="rpg", choices=loss_choices)
    args = parser.parse_args()
    num_players = 2

    env_configs = {"players": num_players}
    env = rl_environment.Environment(args.game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    # pylint: disable=g-complex-comprehension
    agents = [
        policy_gradient.PolicyGradient(
            idx,
            info_state_size,
            num_actions,
            loss_str=args.loss_str,
            hidden_layers_sizes=(128,)) for idx in range(num_players)
    ]
    expl_policies_avg = PolicyGradientPolicies(env, agents)

    here = os.path.dirname(os.path.abspath(__file__))
    directory = f"{here}/../../results"
    directory = directory + f"/nash/{args.game}/"
    Path(directory).mkdir(parents=True, exist_ok=True)
    fn = directory + args.loss_str
    expls = []
    for ep in range(args.num_episodes):
        if (ep + 1) % args.eval_every == 0:
            losses = [agent.loss for agent in agents]
            expl = exploitability.exploitability(env.game, expl_policies_avg)
            expls.append([ep + 1, expl.item()])
            print(ep + 1, expl.item())
        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)
        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)

    expls = np.array(expls)
    df = pd.DataFrame(data={'Exploitability': expls[:, 1]}, index=expls[:, 0].astype(np.int32))
    df.to_csv(fn + '.csv')
    df.plot()
    plt.xlabel("Ep")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(fn + ".png")