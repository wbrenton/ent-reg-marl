import sys
import argparse
from open_spiel.python import rl_environment

if __name__ == "__main__":
    sys.path.append("/Users/will/Home/ML/Projects/soft-dqn/")
    from algorithms.nfsp import nfsp

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="dark_hex")
    parser.add_argument("--eval", default="random_opponent", choices=["exploitability", "random_opponent"])
    parser.add_argument("--exp_name", default="nfsp")
    parser.add_argument("--num_players", default=2)
    parser.add_argument("--num_env_steps", default=10_000_000)
    parser.add_argument("--eval_every", default=100_000)
    parser.add_argument("--replay_buffer_capacity", default=200_000)
    parser.add_argument("--hidden_layer_sizes", default=[512, 512, 512])
    parser.add_argument("--reservoir_buffer_capacity", default=2_000_000)
    args = parser.parse_args()

    env = rl_environment.Environment(args.game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    kwargs = {
        "replay_buffer_capacity": args.replay_buffer_capacity,
        "epsilon_decay_duration": args.num_env_steps,
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
    }

    agents = [
        nfsp.NFSP(
            idx,
            info_state_size,
            num_actions,
            hidden_layers_sizes=args.hidden_layer_sizes,
            reservoir_buffer_capacity=args.reservoir_buffer_capacity,
            **kwargs
        ) for idx in range(args.num_players)
    ]

    if args.eval == "exploitability":
        from algorithms.utils.nash import train_and_evaluate_nash
        train_and_evaluate_nash(args, env, agents)

    elif args.eval == "random_opponent":
        from algorithms.utils.random_opponent import train_and_evaluate_against_random
        train_and_evaluate_against_random(args, env, agents)