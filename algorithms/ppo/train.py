import sys
import argparse
from open_spiel.python import rl_environment

if __name__ == "__main__":
    sys.path.append("/Users/will/Home/ML/Projects/soft-dqn/")
    from algorithms.ppo import ppo
    from algorithms.utils.ppo_nash import train_and_evaluate_nash

    learning_rate = 0.01 # 2.5e-4
    num_steps = 128
    gamma = 1.0
    ent_coef = 0.03125

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42)
    parser.add_argument("--game", default="dark_hex")
    parser.add_argument("--eval", default="random_opponent", choices=["exploitability", "random_opponent"])
    parser.add_argument("--exp_name", default=f"ppo_lr_{learning_rate}_ent_coef_{ent_coef}")
    parser.add_argument("--num_players", default=2)
    parser.add_argument("--num_env_steps", default=10_000_000)
    parser.add_argument("--eval_every", default=100_000)

    parser.add_argument("--learning_rate", default=learning_rate)
    parser.add_argument("--num_steps", default=num_steps)
    parser.add_argument("--anneal_lr", default=False)
    parser.add_argument("--gae", default=True)
    parser.add_argument("--gamma", default=gamma)
    parser.add_argument("--gae_lambda", default=0.95)
    parser.add_argument("--num_minibatches", default=4)
    parser.add_argument("--update_epochs", default=4)
    parser.add_argument("--norm_adv", default=True)
    parser.add_argument("--clip_coef", default=0.1)
    parser.add_argument("--clip_vloss", default=True)
    parser.add_argument("--ent_coef", default=ent_coef)
    parser.add_argument("--vf_coef", default=0.5)
    parser.add_argument("--max_grad_norm", default=0.5)
    parser.add_argument("--target_kl", default=None)

    args = parser.parse_args()

    env = rl_environment.Environment(args.game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    agents = ppo.PPO(
            input_shape=info_state_size,
            num_actions=num_actions,
            num_players=args.num_players,
            steps_per_batch=args.num_steps,
            num_minibatches=args.num_minibatches,
            update_epochs=args.update_epochs,
            learning_rate=args.learning_rate,
            gae=args.gae,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            normalize_advantages=args.norm_adv,
            clip_coef=args.clip_coef,
            clip_vloss=args.clip_vloss,
            entropy_coef=args.ent_coef,
            value_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.target_kl,
            agent_fn=ppo.PPOAgent)

    if args.eval == "exploitability":
        from algorithms.utils.ppo_nash import train_and_evaluate_nash
        train_and_evaluate_nash(args, env, agents)

    elif args.eval == "random_opponent":
        from algorithms.utils.ppo_random_opponent import train_and_evaluate_against_random
        train_and_evaluate_against_random(args, env, agents)