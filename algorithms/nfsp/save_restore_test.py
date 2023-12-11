"""NFSP agents trained on Leduc Poker."""

import copy
from absl import app
from absl import flags
from absl import logging


import jax
import chex
import numpy as np

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
import nfsp # open-spiel/python/jax/nfsp.py with checkpointing implemented

FLAGS = flags.FLAGS

flags.DEFINE_string("game_name", "leduc_poker",
                    "Name of the game.")
flags.DEFINE_integer("num_players", 2,
                     "Number of players.")
flags.DEFINE_integer("num_train_episodes", int(20e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
    128,
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(2e5),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_integer("min_buffer_size_to_learn", 1000,
                     "Number of samples in buffer before learning begins.")
flags.DEFINE_float("anticipatory_param", 0.1,
                   "Prob of using the rl best response as episode policy.")
flags.DEFINE_integer("batch_size", 128,
                     "Number of transitions to sample at each learning step.")
flags.DEFINE_integer("learn_every", 64,
                     "Number of steps between learning updates.")
flags.DEFINE_float("rl_learning_rate", 0.01,
                   "Learning rate for inner rl agent.")
flags.DEFINE_float("sl_learning_rate", 0.01,
                   "Learning rate for avg-policy sl network.")
flags.DEFINE_string("optimizer_str", "sgd",
                    "Optimizer, choose from 'adam', 'sgd'.")
flags.DEFINE_string("loss_str", "mse",
                    "Loss function, choose from 'mse', 'huber'.")
flags.DEFINE_integer("update_target_network_every", 19200,
                     "Number of steps between DQN target network updates.")
flags.DEFINE_float("discount_factor", 1.0,
                   "Discount factor for future rewards.")
flags.DEFINE_integer("epsilon_decay_duration", int(20e6),
                     "Number of game steps over which epsilon is decayed.")
flags.DEFINE_float("epsilon_start", 0.06,
                   "Starting exploration parameter.")
flags.DEFINE_float("epsilon_end", 0.001,
                   "Final exploration parameter.")
flags.DEFINE_string("evaluation_metric", "nash_conv",
                    "Choose from 'exploitability', 'nash_conv'.")
flags.DEFINE_bool("use_checkpoints", True, "Save/load neural network weights.")
flags.DEFINE_string("checkpoint_dir", "/admin/home-willb/gtrl/tmp/nfsp_leduc_poker",
                    "Directory to save/load the agent.")

def main(unused_argv):
    logging.info("Loading %s", FLAGS.game_name)
    game = FLAGS.game_name
    num_players = FLAGS.num_players

    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    # initialize agent and set params to zero then save
    save_agent = nfsp.NFSP(0, info_state_size, num_actions)
    save_agent._rl_agent.params_q_network = jax.tree_map(lambda x: x * 0, save_agent._rl_agent.params_q_network)
    save_agent.params_avg_network = jax.tree_map(lambda x: x * 0, save_agent.params_avg_network)
    
    save_agent.save(FLAGS.checkpoint_dir)
    
    agent = nfsp.NFSP(0, info_state_size, num_actions)
    init_agent = copy.deepcopy(agent)
    if not agent.has_checkpoint(FLAGS.checkpoint_dir):
        raise ValueError("Checkpoint not found in %s" % FLAGS.checkpoint_dir)

    agent.restore(FLAGS.checkpoint_dir)
    
    chex.assert_trees_all_close(agent._rl_agent.params_q_network, save_agent._rl_agent.params_q_network)
    
    # this should fail
    chex.assert_trees_all_close(agent.params_avg_network, init_agent.params_avg_network)


if __name__ == "__main__":
    app.run(main)
