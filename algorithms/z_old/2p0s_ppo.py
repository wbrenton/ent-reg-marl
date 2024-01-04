import sys
sys.path.append('/Users/will/Home/ML/Projects/soft-dqn/')

import os
import jax
# import tyro
from dataclasses import dataclass, field
from typing import List
import numpy as np

from open_spiel.python.algorithms import random_agent
from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
from algorithms.ppo import ppo
from rich.pretty import pprint

from typing import Optional
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    # General experiment parameters
    exp_name: str = field(default_factory=lambda: os.path.basename(__file__).rstrip(".py"))
    "The name of this experiment."
    game_name: str = "phantom_ttt"
    "The id of the OpenSpiel game."
    num_players: int = 2
    "Number of players."
    num_train_episodes: int = int(20e6)
    "Number of training episodes."
    num_train_environment_steps: int = int(10e6)
    "Number of training environment steps."
    eval_every: int = 1000
    "Episode frequency at which the agents are evaluated."
    use_checkpoints: bool = True
    "Save/load neural network weights."
    checkpoint_dir: str = "models/"
    "Directory to save/load the agent."
    
    learning_rate: float = 2.5e-4
    "The learning rate of the optimizer."
    seed: int = 1
    "Seed of the experiment."
    total_timesteps: int = 10_000_000
    "Total timesteps of the experiments."
    eval_every: int = 10
    "Evaluate the policy every N updates."
    torch_deterministic: bool = True
    "If toggled, `torch.backends.cudnn.deterministic=False`."
    cuda: bool = True
    "If toggled, cuda will be enabled by default."

    # Algorithm specific arguments
    num_envs: int = 8
    "The number of parallel game environments."
    num_steps: int = 128
    "The number of steps to run in each environment per policy rollout."
    anneal_lr: bool = True
    "Toggle learning rate annealing for policy and value networks."
    gae: bool = True
    "Use GAE for advantage computation."
    gamma: float = 0.99
    "The discount factor gamma."
    gae_lambda: float = 0.95
    "The lambda for the general advantage estimation."
    num_minibatches: int = 4
    "The number of mini-batches."
    update_epochs: int = 4
    "The K epochs to update the policy."
    norm_adv: bool = True
    "Toggles advantages normalization."
    clip_coef: float = 0.1
    "The surrogate clipping coefficient."
    clip_vloss: bool = True
    "Toggles whether or not to use a clipped loss for the value function."
    ent_coef: float = 0.05
    "Coefficient of the entropy."
    vf_coef: float = 0.5
    "Coefficient of the value function."
    max_grad_norm: float = 0.5
    "The maximum norm for the gradient clipping."
    target_kl: Optional[float] = None
    "The target KL divergence threshold."

    # Evaluation
    evaluation_opponent: str = 'random'
    "Choose from 'random', 'previous', ''."
    eval_episodes: int = 1000
    "How many episodes to run per eval."

def eval_against_fixed_bots(env, trained_agents, fixed_agents, num_episodes):
  """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
  num_players = len(fixed_agents)
  sum_episode_rewards = np.zeros(num_players)
  for player_pos in range(num_players):
    cur_agents = fixed_agents[:]
    cur_agents[player_pos] = trained_agents
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
          action_list = [output.action for output in agent_output] if player_pos == player_id else [agent_output.action]
        else:
          agents_output = [
              agent.step(time_step, is_evaluation=True) for agent in cur_agents
          ]
          action_list = [agent_output.action for agent_output in agents_output]
        time_step = env.step(action_list)
        episode_rewards += time_step.rewards[player_pos]
      sum_episode_rewards[player_pos] += episode_rewards
  return sum_episode_rewards / num_episodes

def evalution(env, agents, writer, global_step):
    if args.evaluation_opponent == "random":
        eval_agents = [
            random_agent.RandomAgent(player_id=idx, num_actions=num_actions)
            for idx in range(num_players)
            ]
    # elif args.evaluation_opponent == 'previous':
    #     eval_agents = [
    #         soft_dqn.SoftDQN(idx, info_state_size, num_actions, args.alpha, hidden_layers_sizes,
    #                     ) for idx in range(num_players)
    #     ]
    #     for agent in eval_agents:
    #         if agent.has_checkpoint(checkpoint_dir):
    #             agent.restore(checkpoint_dir)
    else:
        raise ValueError("Invalid evaluation opponent, choose from 'random', 'previous'.")
    
    reward_mean = eval_against_fixed_bots(env, agents, eval_agents, args.eval_episodes)
    writer.add_scalar("charts/mean_reward_p0", reward_mean[0], global_step)
    writer.add_scalar("charts/mean_reward_p1", reward_mean[1], global_step)

    print()
    for i, reward in enumerate(reward_mean):
        print(f"Player Position: {i} Reward: {reward} Loss: {agent.loss}")

    if args.use_checkpoints:
        agent.save(checkpoint_dir)
    print("_____________________________________________", flush=True)

if __name__ == "__main__":
    args = Args()
    pprint(args)
    print(f"Loading {args.game_name}")
    game = args.game_name
    num_players = args.num_players
    batch_size = int(args.num_envs * args.num_steps)
    run_name = f"ppo-game~{game}-ent_coef~{args.ent_coef}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
      "hyperparameters",
      "|param|value|\n|-|-|\n%s" %
      ("\n".join([f"|{key}|{value}|" for key, value in vars(Args).items()])),
    )
    checkpoint_dir = args.checkpoint_dir + run_name
    print(run_name)

    env_configs = {"players": num_players}
    if game in ["leduc_poker", "kuhn_poker"]:
        env = rl_environment.Environment(game, **env_configs)
    elif game in ["dark_hex", "phantom_ttt"]:  # these don't have num_players args
        env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agent = ppo.PPO(
        input_shape=(info_state_size,),
        num_actions=num_actions,
        num_players=num_players,
        agent_fn=ppo.PPOAgent,
        writer=writer,
    )

    # if args.use_checkpoints:
    #     for agent in agents:
    #         if agent.has_checkpoint(checkpoint_dir):
    #             agent.restore(checkpoint_dir)

    print(f"Training PPO on: {jax.devices()}", flush=True)
    num_updates = args.total_timesteps // batch_size
    time_step = env.reset()
    for update in range(num_updates):
        for _ in range(args.num_steps):
            current_player = time_step.observations["current_player"]
            agent_output = agent.step(time_step)
            action_list = [output.action for output in agent_output]
            time_step = env.step(action_list)
            reward = time_step.rewards[current_player]
            reward = reward if reward is not None else 0.
            done = time_step.last()
            agent.post_step(reward, done)

            if done:
                # agent.step(time_step)
                time_step = env.reset()

        if args.anneal_lr:
            agent.anneal_learning_rate(update // num_players, num_updates)

        agent.learn(time_step)

        if (update + 1) % args.eval_every == 0:
            print((update + 1) * batch_size, "environment steps")
            evalution(env, agent, writer, (update + 1) * batch_size)
            time_step = env.reset()