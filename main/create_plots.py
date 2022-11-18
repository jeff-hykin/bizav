import sys
from blissful_basics import FS, print
from main.utils import LogReader, LiquidLog, plot_line
from informative_iterator import ProgressBar

log_reader = LogReader()
@log_reader.add_preprocessor()
def hyperparam_logs(string):
    lines = []
    for _, line in ProgressBar(string.splitlines(), title="parsing hyperparams"):
        if "}. Best is trial" in line and "and parameters: {" in line:
            start = line.index("and parameters: {") + len("and parameters: {") - 1
            end = line.index("}. Best is trial") + 1
            line = line[start:end].replace("None", "null").replace("True", 'true').replace("False", 'false').replace("'", '"')
        lines.append(line)
    return '\n'.join(lines)

@log_reader.add_preprocessor()
def bandit_logs(string):
    lines = []
    for _, line in ProgressBar(string.splitlines(), title="parsing ucb output"):
        # Step 50 3 visits [1.0, 3.0, 1.0, 34.0, 6.0, 1.0, 4.0]  episode_count: 31 q_vals: [-11.111, -9.362, -11.111, -8.611, -9.259, -11.111, -9.539]
        if line.startswith("Step ") and "episode_count:" in lines and "q_vals:" in line:
            step          = line[len("Step"):line.index("visits")].split()[0]
            visits        = line[line.index("visits"):line.index("episode_count:")]
            episode_count = line[line.index("episode_count:"):line.index("q_vals:")]
            q_vals        = line[line.index("q_vals:")+len("q_vals:"):]
            line = f'{{"step":{step}, "visits": {visits}, "episode_count":{episode_count}, "q_vals": {q_vals}}}'
        lines.append(line)
    return '\n'.join(lines)

@log_reader.add_preprocessor()
def final_evals(string):
    lines = [
        '{"final_eval":false}'
    ]
    for _, line in ProgressBar(string.splitlines(), title="parsing final_evals"):
        # Step 50 3 visits [1.0, 3.0, 1.0, 34.0, 6.0, 1.0, 4.0]  episode_count: 31 q_vals: [-11.111, -9.362, -11.111, -8.611, -9.259, -11.111, -9.539]
        if line.startswith("final_eval: "):
            line = line[len("final_eval: "):].replace("None", "null").replace("True", 'true').replace("False", 'false').replace("'", '"').replace("}", ', "final_eval": true }')
        lines.append(line)
    return '\n'.join(lines)

# 
# read all the files, compile them into one big frame
# 
print("reading files")
ProgressBar.seconds_per_print = 0.08
with print.indent:
    liquid_log = LiquidLog()
    for progress, each in ProgressBar(sys.argv[1:]):
        liquid_log = liquid_log.concat(
            LiquidLog(log_reader.read(each, disable_logging=False))
        )
        progress.pretext = "Finished "+FS.basename(each)

# 
# add data
# 
liquid_log = liquid_log.map(
    lambda row: ({
        **row,
        "env": FS.basename(row["__source__"]).split('__')[0],
        "atk": FS.basename(row["__source__"]).split('__')[1].split('=')[1],
        "def": FS.basename(row["__source__"]).split('__')[2].split('=')[1].replace(".log", ""),
    })
)

from super_hash import super_hash
def not_a_duplicate():
    values = set()
    def checker(row):
        hash_value = super_hash(row)
        if hash_value in values:
            return False
        else:
            values.add(hash_value)
            return True
    return checker

with print.indent:
    for env_name, each_env_frame in liquid_log.grouped_by('env').items():
        for (attack_name, defence_name), each_scenario_frame in each_env_frame.grouped_by('atk', 'def').items():
            print(f'''(attack_name, defence_name) = {(attack_name, defence_name)}''')
            
            # 
            # plot training curve
            # 
            training_logs = each_scenario_frame.only_keep_if(
                lambda row: row["total_number_of_episodes"] and row['per_episode_reward'] != None
            ).only_keep_if(not_a_duplicate()).sort_by('total_number_of_episodes')
            first_element = next(iter(training_logs))
            folder = FS.basename(FS.dirname(first_element['__source__']))
            plot_line(
                plot_name=f"{folder}/train/{env_name}",
                line_name=f"{attack_name}_{defence_name}",
                new_x_values=training_logs['total_number_of_episodes'],
                new_y_values=training_logs['per_episode_reward'],
            )
