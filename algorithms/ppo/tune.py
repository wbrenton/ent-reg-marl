import sys
import argparse
from open_spiel.python import rl_environment

def train(learning_rate: float, num_steps: int, anneal_lr: bool, gamma: float, ent_coef: float):
    sys.path.append("/Users/will/Home/ML/Projects/soft-dqn/")
    from algorithms.ppo import ppo
    from algorithms.utils.ppo_nash import train_and_evaluate_nash

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42)
    parser.add_argument("--game", default="kuhn_poker")
    parser.add_argument("--exp_name", default=f"ppo_lr_{learning_rate}_ent_coef_{ent_coef}")
    parser.add_argument("--num_players", default=2)
    parser.add_argument("--num_env_steps", default=10_000_000)
    parser.add_argument("--eval_every", default=10_000)

    parser.add_argument("--learning_rate", default=learning_rate)
    parser.add_argument("--num_steps", default=num_steps)
    parser.add_argument("--anneal_lr", default=anneal_lr)
    parser.add_argument("--gae", default=True)
    parser.add_argument("--gamma", default=gamma)
    parser.add_argument("--gae_lambda", default=0.95) # TODO: try 1.0
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
    print("-"*100)
    print(args.exp_name)

    env_configs = {"players": args.num_players}
    env = rl_environment.Environment(args.game, **env_configs)
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

    return train_and_evaluate_nash(args, env, agents)

if __name__ == "__main__":
    import optuna

    def objective(trial):
        learning_rate = trial.suggest_categorical("learning_rate", [2.5e-4, 0.1e-4])
        # num_steps = trial.suggest_categorical("num_steps", [1, 2, 4, 8, 16, 32, 64, 128])
        num_steps = 128
        # anneal_lr = trial.suggest_categorical("anneal_lr", [False, True])
        anneal_lr = False
        # gamma = trial.suggest_categorical("gamma", [0.99, 1.0])
        gamma = 1.0
        ent_coef = trial.suggest_categorical("ent_coef", [0.015625, 0.03125, 0.05, 0.0625, 0.125])

        return train(learning_rate, num_steps, anneal_lr, gamma, ent_coef)

    study = optuna.create_study(
        sampler=optuna.samplers.TPESampler(seed=42),
        direction='minimize'
        )

    study.optimize(objective, n_trials=1000)