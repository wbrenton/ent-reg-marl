import sys
import argparse
from open_spiel.python import rl_environment

if __name__ == "__main__":
    sys.path.append("/Users/will/Home/ML/Projects/soft-dqn/")
    from algorithms.policy_gradient import policy_gradient
    from algorithms.utils.nash import train_and_evaluate_nash

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="kuhn_poker")
    parser.add_argument("--exp_name", default="a2c")
    parser.add_argument("--num_players", default=2)
    parser.add_argument("--num_env_steps", default=10_000_000)
    parser.add_argument("--eval_every", default=10_000)
    args = parser.parse_args()

    env_configs = {"players": args.num_players}
    env = rl_environment.Environment(args.game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agents = [
        policy_gradient.PolicyGradient(
            idx,
            info_state_size,
            num_actions,
            loss_str="a2c",
            hidden_layers_sizes=(128,)) for idx in range(args.num_players)
    ]

    train_and_evaluate_nash(args, env, agents)