"""RL agents trained against fixed policy/bot as approximate responses.

This can be used to try to find exploits in policies or bots, as described in
Timbers et al. '20 (https://arxiv.org/abs/2004.09677), but only using RL
directly rather than RL+Search.
"""

import tyro
import numpy as np
from typing import Literal, List
from dataclasses import dataclass, field

from open_spiel.python import rl_agent
from open_spiel.python import rl_environment
from open_spiel.python import rl_tools
from open_spiel.python.jax import dqn
from open_spiel.python.algorithms import random_agent
from open_spiel.python.algorithms import tabular_qlearner
import soft_dqn

@dataclass
class TrainingConfig:
    # Training parameters
    checkpoint_dir: str = "/tmp/dqn_br/"
    "Directory to save/load the agent models."
    exploitee_checkpoint_dir: str = "/admin/home-willb/gtrl/tmp/"
    "Directory to load the exploitees."
    num_train_episodes: int = int(1e6)
    "Number of training episodes."
    eval_every: int = 1000
    "Episode frequency at which the DQN agents are evaluated."
    eval_episodes: int = 1000
    "How many episodes to run per eval."

    # DQN model hyper-parameters
    hidden_layers_sizes: List[int] = field(default_factory=lambda: [64, 64, 64])
    "Number of hidden units in the Q-Network MLP."
    replay_buffer_capacity: int = int(1e5)
    "Size of the replay buffer."
    batch_size: int = 32
    "Number of transitions to sample at each learning step."

    # Exploitee hyper-parameters
    optimizer_str: str = "adam"
    "Optimizer, choose from 'adam', 'sgd'."
    learning_rate: float = 0.01
    "Learning rate for exploitee agent."
    alpha: float = 0.05
    "Softmax temperature for inner rl agent."

    # Main algorithm parameters
    seed: int = 0
    "Seed to use for everything."
    window_size: int = 30
    "Size of window for rolling average."
    num_players: int = 2
    "Number of players."
    game_name: str = "dark_hex"
    "Game string."
    exploitee: str = "soft_dqn"
    "Exploitee (random | first | soft_dqn)."
    learner: str = "dqn"
    "Learner (qlearning | dqn)."

def eval_against_fixed_bots(env, trained_agents, fixed_agents, num_episodes):
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


def create_training_agents(num_players, num_actions, info_state_size,
                           hidden_layers_sizes):
  """Create the agents we want to use for learning."""
  if args.learner == "qlearning":
    # pylint: disable=g-complex-comprehension
    return [
        tabular_qlearner.QLearner(
            player_id=idx,
            num_actions=num_actions,
            # step_size=0.02,
            step_size=0.1,
            # epsilon_schedule=rl_tools.ConstantSchedule(0.5),
            epsilon_schedule=rl_tools.LinearSchedule(0.5, 0.2, 1000000),
            discount_factor=0.99) for idx in range(num_players)
    ]
  elif args.learner == "dqn":
    # pylint: disable=g-complex-comprehension
    return [
        dqn.DQN(
            player_id=idx,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            discount_factor=0.99,
            epsilon_start=0.5,
            epsilon_end=0.1,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=args.replay_buffer_capacity,
            batch_size=args.batch_size) for idx in range(num_players)
    ]
  else:
    raise RuntimeError("Unknown learner")


class FirstActionAgent(rl_agent.AbstractAgent):
  """An example agent class."""

  def __init__(self, player_id, num_actions, name="first_action_agent"):
    assert num_actions > 0
    self._player_id = player_id
    self._num_actions = num_actions

  def step(self, time_step, is_evaluation=False):
    # If it is the end of the episode, don't select an action.
    if time_step.last():
      return

    # Pick the first legal action.
    cur_legal_actions = time_step.observations["legal_actions"][self._player_id]
    action = cur_legal_actions[0]
    probs = np.zeros(self._num_actions)
    probs[action] = 1.0

    return rl_agent.StepOutput(action=action, probs=probs)


class RollingAverage(object):
  """Class to store a rolling average."""

  def __init__(self, size=100):
    self._size = size
    self._values = np.array([0] * self._size, dtype=np.float64)
    self._index = 0
    self._total_additions = 0

  def add(self, value):
    self._values[self._index] = value
    self._total_additions += 1
    self._index = (self._index + 1) % self._size

  def mean(self):
    n = min(self._size, self._total_additions)
    if n == 0:
      return 0
    return self._values.sum() / n


if __name__ == "__main__":
    args = tyro.cli(TrainingConfig)
    exploitee_run_name = f"soft_dqn-game~{args.game_name}-opt~{args.optimizer_str}-lr~{args.learning_rate}-alpha~{args.alpha}"
    exploitee_checkpoint_dir = args.exploitee_checkpoint_dir + exploitee_run_name
    np.random.seed(args.seed)

    num_players = args.num_players

    env = rl_environment.Environment(args.game_name, include_full_state=True)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    # Exploitee agents
    if args.exploitee == "first":
        exploitee_agents = [
            FirstActionAgent(idx, num_actions) for idx in range(num_players)
        ]
    elif args.exploitee == "random":
        exploitee_agents = [
            random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
            for idx in range(num_players)
        ]
    elif args.exploitee == "soft_dqn":
        exploitee_agents = [
            soft_dqn.SoftDQN(idx, info_state_size, num_actions)
            for idx in range(num_players)
        ]
        for agent in exploitee_agents:
            if agent.has_checkpoint(exploitee_checkpoint_dir):
                agent.restore(exploitee_checkpoint_dir)
            else:
              raise RuntimeError("No exploitee checkpoint found")
    else:
        raise RuntimeError("Unknown exploitee")

    rolling_averager = RollingAverage(args.window_size)
    rolling_averager_p0 = RollingAverage(args.window_size)
    rolling_averager_p1 = RollingAverage(args.window_size)
    rolling_value = 0
    total_value = 0
    total_value_n = 0

    hidden_layers_sizes = [int(l) for l in args.hidden_layers_sizes]
    # pylint: disable=g-complex-comprehension
    learning_agents = create_training_agents(num_players, num_actions,
                                                info_state_size,
                                                hidden_layers_sizes)

    print(exploitee_run_name)
    print("Starting...")

    for ep in range(args.num_train_episodes):
        # evaluate against fixed bots
        if (ep + 1) % args.eval_every == 0:
            r_mean = eval_against_fixed_bots(env, learning_agents, exploitee_agents,
                                                args.eval_episodes)
            value = r_mean[0] + r_mean[1]
            rolling_averager.add(value)
            rolling_averager_p0.add(r_mean[0])
            rolling_averager_p1.add(r_mean[1])
            rolling_value = rolling_averager.mean()
            rolling_value_p0 = rolling_averager_p0.mean()
            rolling_value_p1 = rolling_averager_p1.mean()
            total_value += value
            total_value_n += 1
            avg_value = total_value / total_value_n
            print(f"[{ep + 1}] Mean episode rewards {r_mean}, value: {value:.3f}, "
                f"rval: {rolling_value:.3f} (p0/p1: {rolling_value_p0:.3f} / {rolling_value_p1:.3f}), aval: {avg_value:.3f}", flush=True)


        agents_round1 = [learning_agents[0], exploitee_agents[1]]
        agents_round2 = [exploitee_agents[0], learning_agents[1]]

        for agents in [agents_round1, agents_round2]:
            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if env.is_turn_based:
                    agent_output = agents[player_id].step(time_step)
                    action_list = [agent_output.action]
                else:
                    agents_output = [agent.step(time_step) for agent in agents]
                    action_list = [
                        agent_output.action for agent_output in agents_output
                    ]
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)