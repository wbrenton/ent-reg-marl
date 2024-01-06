import os
import subprocess

SBATCH_SCRIPT = "run.batch"
RESULTS_DIR = "results"
TRAIN_SCRIPT = "algorithms/utils/rl_response.py"

def train_best_responder(args: dict):
    cmd_list = ['sbatch', SBATCH_SCRIPT, TRAIN_SCRIPT] + [item for k, v in args.items() for item in [f"--{k}", str(v)]]
    print(cmd_list)
    result = subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        print("STDOUT:", result.stdout)
    else:
        print("STDERR:", result.stderr)

games = ['phantom_ttt', 'dark_hex']
args = {}


experiments = os.listdir(RESULTS_DIR)
for experiment in experiments:
    algorithm_str = experiment.split('_')[0]
    if 'nfsp' in algorithm_str:
        continue
    for game_str in games:
        args['game'] = game_str
        args['exploitee_path'] = os.path.join(RESULTS_DIR, experiment, game_str) + '/'
        train_best_responder(args)