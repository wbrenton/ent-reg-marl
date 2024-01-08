import sys
import torch
from rich.pretty import pprint
import argparse
from open_spiel.python import rl_environment
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    sys.path.append("/admin/home-willb/soft-dqn/")
    from algorithms.ppo import ppo
    from algorithms.utils.ppo_random_opponent import train_and_evaluate_against_random

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="phantom_ttt")
    parser.add_argument("--exp_name", default="ppo")
    parser.add_argument("--num_players", default=2)
    parser.add_argument("--num_env_steps", default=10_000_000)
    parser.add_argument("--eval_every", default=100_000)
    parser.add_argument("--steps_per_batch", default=128)
    parser.add_argument("--learning_rate", default=2.5e-4, type=float)
    parser.add_argument("--gamma", default=1.0)
    parser.add_argument("--entropy_coef", default=0.05, type=float)
    args = parser.parse_args()
    args.exp_name = f"{args.exp_name}_lr_{args.learning_rate}_alpha_{args.entropy_coef}"
    pprint(args)
    print("CUDA available:", torch.cuda.is_available())

    env = rl_environment.Environment(args.game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    writer = SummaryWriter(f"runs/{args.exp_name}_{args.game}")

    agent = ppo.PPO(
        info_state_size,
        num_actions,
        args.num_players,
        learning_rate=args.learning_rate,
        device="cuda" if torch.cuda.is_available() else "cpu",
        writer=writer,
    )

    train_and_evaluate_against_random(args, env, agent)