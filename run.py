import subprocess

SCRIPT_PATH = "run.batch"
make_train_path = lambda algorithm_str: f"algorithms/{algorithm_str}/train.py"

def train_agent(algorithm_str: str, games: list, args: dict):
    train_path = make_train_path(algorithm_str)
    for game_str in games:
        args['game'] = game_str
        cmd_list = ['sbatch', SCRIPT_PATH, train_path] + [item for k, v in args.items() for item in [f"--{k}", str(v)]]
        print(cmd_list)
        result = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            print("STDOUT:", result.stdout)
        else:
            print("STDERR:", result.stderr)

games = ['phantom_ttt', 'dark_hex']
args = {
    'game':  None,
    'learning_rate': None,
    'entropy_coef': None,
}

lrs = [1e-4, 2.5e-4]
alphas = [0.015625, 0.03125, 0.05, 0.0625, 0.125]
for lr in lrs:
    args['learning_rate'] = lr
    for alpha in alphas:
        args['entropy_coef'] = alpha
        train_agent('ppo', games, args)