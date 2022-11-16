import sys
from blissful_basics import FS
from main.utils import HogLog

log_reader = HogLog()
@log_reader.add_preprocessor()
def hyperparam_logs(string):
    lines = []
    for line in string.splitlines():
        if "}. Best is trial" in line and "and parameters: {" in line:
            start = line.index("and parameters: {") + len("and parameters: {") - 1
            end = line.index("}. Best is trial") + 1
            line = line[start:end].replace("None", "null").replace("True", 'true').replace("False", 'false').replace("'", '"')
        lines.append(line)
    return '\n'.join(lines)

@log_reader.add_preprocessor()
def bandit_logs(string):
    lines = []
    for line in string.splitlines():
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
def bandit_logs(string):
    lines = [
        '{"final_eval":false}'
    ]
    for line in string.splitlines():
        # Step 50 3 visits [1.0, 3.0, 1.0, 34.0, 6.0, 1.0, 4.0]  episode_count: 31 q_vals: [-11.111, -9.362, -11.111, -8.611, -9.259, -11.111, -9.539]
        if line.startswith("final_eval: "):
            line = line[len("final_eval: "):].replace("None", "null").replace("True", 'true').replace("False", 'false').replace("'", '"').replace("}", ', "final_eval": true }')
        lines.append(line)
    return '\n'.join(lines)

for each in sys.argv[1:]:
    log_reader.read(each)


# each = "logs/comparisons1/cartpole__atk=act__def=none.log"
# log_reader.read("logs/comparisons1/cartpole__atk=act__def=none.log")

import pandas 
df = pandas.DataFrame(log_reader.frame)

from main.utils import plot_line

for each in sys.argv[1:]:
    this_df = df.copy()
    this_df = this_df[this_df["__source__"] == each]
    this_df = this_df[this_df["total_number_of_episodes"] == this_df["total_number_of_episodes"]]
    plot_line(
        plot_name="default",
        line_name=FS.basename(each),
        new_x_values=this_df['total_number_of_episodes'].tolist(),
        new_y_values=this_df['per_episode_reward'].tolist()
    )
    print(f'''this_df[df["final_eval"] != None] = {df[df["final_eval"] == df["final_eval"]][df["final_eval"] == True]}''')