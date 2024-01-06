import os
import pandas as pd
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
result_dirs = os.listdir(current_dir)

games = ['dark_hex', 'phantom_ttt']

for game in games:
    sums = []
    pids_0 = []
    pids_1 = []
    names = []
    for result in result_dirs:
        algorithm_str = result.split('_')[0]
        if algorithm_str not in ['nfsp', 'ppo']:
            continue
        result_game_dir = os.path.join(current_dir, result, game)
        csv_file = os.path.join(result_game_dir, '_vs_random.csv')
        df = pd.read_csv(csv_file, index_col=0)
        # sum pid 0 and 1
        df[f'{result}'] = df['pid_0'] + df['pid_1']
        sums.append(df[[f'{result}']])
        pids_0.append(df[['pid_0']])
        pids_1.append(df[['pid_1']])
        names.append(result)

    plt.figure(figsize=(100, 5))

    # plot sums
    current_dir += '/'
    df = pd.concat(sums, axis=1)
    df.to_csv(current_dir + 'sum.csv')
    df = df.rolling(10).mean()
    df.plot(figsize=(20, 10))
    plt.xlabel("Environment Steps")
    plt.ylabel("pid_0 + pid_1")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend to the right
    plt.subplots_adjust(right=0.75)  # Adjust subplot to fit the legend
    plt.savefig(current_dir + "sum.png")
    
    # plot pid 0
    df = pd.concat(pids_0, axis=1)
    df.columns = [f'{name}_pid_1' for name in names]
    df.to_csv(current_dir + 'pid_0.csv')
    df = df.rolling(10).mean()
    df.plot(figsize=(20, 10))
    plt.xlabel("Environment Steps")
    plt.ylabel("pid_0")
    plt.grid(True)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend to the right
    plt.subplots_adjust(right=0.75)  # Adjust subplot to fit the legend
    plt.savefig(current_dir + "pid_0.png")
    
    # plot pid 1
    df = pd.concat(pids_1, axis=1)
    df.columns = [f'{name}_pid_1' for name in names]
    df.to_csv(current_dir + 'pid_1.csv')
    df = df.rolling(10).mean()
    df.plot(figsize=(20, 10))
    plt.xlabel("Environment Steps")
    plt.ylabel("pid_1")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend to the right
    plt.subplots_adjust(right=0.75)  # Adjust subplot to fit the legend
    plt.savefig(current_dir + "pid_1.png")
