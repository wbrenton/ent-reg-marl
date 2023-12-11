"""NFSP agents trained on Leduc Poker."""

import jax
import tyro
from dataclasses import dataclass, field
from typing import List

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
import soft_dqn # open-spiel/python/jax/nfsp.py with checkpointing implemented

@dataclass
class TrainingConfig:
    # Training parameters
    game_name: str = "dark_hex"
    "Name of the game."
    num_players: int = 2
    "Number of players."
    num_train_episodes: int = int(20e6)
    "Number of training episodes."
    num_train_environment_steps: int = int(10e6)
    "Number of training environment steps."
    eval_every: int = 10000
    "Episode frequency at which the agents are evaluated."
    checkpoint_dir: str = "/admin/home-willb/gtrl/tmp/"
    "Directory to save/load the agent."

    # DQN model hyper-parameters
    hidden_layers_sizes: List[int] = field(default_factory=lambda: [128])
    "Number of hidden units in the avg-net and Q-net."
    replay_buffer_capacity: int = 10000
    "Size of the replay buffer."
    batch_size: int = 128
    "Number of transitions to sample at each learning step."
    update_target_network_every: int = 1000
    "Number of steps between DQN target network updates."
    discount_factor: float = 1.0
    "Discount factor for future rewards."
    use_checkpoints: bool = True
    "Save/load neural network weights."

    # Training algorithm parameters
    alpha: float = 0.05
    "Size of the replay buffer."
    min_buffer_size_to_learn: int = 1000
    "Number of samples in buffer before learning begins."
    learn_every: int = 64
    "Number of steps between learning updates."
    learning_rate: float = 0.01
    "Learning rate for inner rl agent."
    optimizer_str: str = "sgd"
    "Optimizer, choose from 'adam', 'sgd'."
    loss_str: str = "mse"
    "Loss function, choose from 'mse', 'huber'."
    evaluation_metric: str = ''
    "Choose from 'exploitability', 'nash_conv', ''."



class NFSPPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies, mode):
    game = env.game
    player_ids = list(range(args.num_players))
    super(NFSPPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._mode = mode
    self._obs = {
        "info_state": [None] * args.num_players,
        "legal_actions": [None] * args.num_players
    }

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

if __name__ == "__main__":
    args = tyro.cli(TrainingConfig)
    print(f"Loading {args.game_name}")
    game = args.game_name
    num_players = args.num_players
    run_name = f"soft_dqn-game~{game}-opt~{args.optimizer_str}-lr~{args.learning_rate}-alpha~{args.alpha}"
    checkpoint_dir = args.checkpoint_dir + run_name
    print(run_name)

    env_configs = {"players": num_players}
    if game in ["leduc_poker", "kuhn_poker"]:
        env = rl_environment.Environment(game, **env_configs)
    elif game in ["dark_hex", "phantom_ttt"]:  # these don't have num_players args
        env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [int(l) for l in args.hidden_layers_sizes]

    agents = [
        soft_dqn.SoftDQN(idx, info_state_size, num_actions, args.alpha, hidden_layers_sizes,
                    ) for idx in range(num_players)
    ]

    if args.use_checkpoints:
        for agent in agents:
            if agent.has_checkpoint(checkpoint_dir):
                agent.restore(checkpoint_dir)

    print(f"Training Soft-DQN on: {jax.devices()}", flush=True)
    for ep in range(args.num_train_episodes):
        if (ep + 1) % args.eval_every == 0:
            losses = [agent.loss for agent in agents]
            print(f"Losses: {losses}")
            if args.evaluation_metric == "exploitability":
                expl = exploitability.exploitability(env.game, joint_avg_policy)
                print(f"[{ep + 1}] Exploitability AVG {expl}")
            elif args.evaluation_metric == "nash_conv":
                nash_conv = exploitability.nash_conv(env.game, joint_avg_policy)
                print(f"[{ep + 1}] NashConv {nash_conv}")
            elif args.evaluation_metric == '':
                pass
            else:
                raise ValueError("Invalid evaluation metric, choose from 'exploitability', 'nash_conv'.")

            if args.use_checkpoints:
                for agent in agents:
                    agent.save(checkpoint_dir)
            print("_____________________________________________", flush=True)

        time_step = env.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]
            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]
            time_step = env.step(action_list)

        for agent in agents:
            agent.step(time_step)
        
        if agents[0].step_counter >= args.num_train_environment_steps:
            print(f"Training complete. Steps: {agents[0].step_counter}")
            break

