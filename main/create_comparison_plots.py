import sys
from blissful_basics import FS, print
from main.utils import log_reader, LiquidLog, plot_line
from informative_iterator import ProgressBar

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
        "def": FS.basename(row["__source__"]).split('__')[2].split('=')[1].replace(".liquid.log", "").replace(".log", ""),
        "total_number_of_episodes": row.get("total_number_of_episodes", row.get("number_of_episodes", None)),
    })
)

print("creating plots")
with print.indent:
    for env_name, each_env_frame in liquid_log.grouped_by('env').items():
        for (attack_name, defence_name), each_scenario_frame in each_env_frame.grouped_by('atk', 'def').items():
            line_frames = each_scenario_frame.split(
                is_divider=lambda row: row["final_eval"]
            )
            print(f'''attack={attack_name} defence={defence_name}''')
            with print.indent:
                for index, each_run_frame in enumerate(line_frames):
                    # 
                    # plot training curve
                    # 
                    training_logs = each_run_frame.only_keep_if(
                        lambda row: row["total_number_of_episodes"] and row['per_episode_reward'] != None
                    ).sort_by('total_number_of_episodes')
                    if len(training_logs) > 0:
                        print(f'''(attack_name, defence_name, index) = {(attack_name, defence_name, index)}''')
                        first_element = next(iter(training_logs))
                        folder = FS.basename(FS.dirname(first_element['__source__']))
                        plot_line(
                            plot_name=f"{folder}/train/{env_name}",
                            line_name=f"{attack_name}_{defence_name}_{index}",
                            new_x_values=training_logs['total_number_of_episodes'],
                            new_y_values=training_logs['per_episode_reward'],
                        )
