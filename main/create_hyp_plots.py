import sys
from blissful_basics import FS, print
from main.utils import log_reader, LiquidLog, plot_line
from informative_iterator import ProgressBar

file_path = sys.argv[1:][0]

@log_reader.add_preprocessor()
def final_evals(string):
    return string.replace("args = {", '{ "start_of_run": true }')

# 
# read all the files, compile them into one big frame
# 
liquid_log = LiquidLog(log_reader.read(file_path, disable_logging=False))

# this split doesnt 100% work but whatever
line_frames = liquid_log.split(
    is_divider=lambda row: row["permaban_threshold"]
)
with print.indent:
    for index, each_run_frame in enumerate(line_frames):
        # 
        # plot training curve
        # 
        training_logs = each_run_frame.only_keep_if(
            lambda row: row["total_number_of_episodes"] and row['per_episode_reward'] != None
        ).sort_by('total_number_of_episodes')
        if len(training_logs) > 0:
            first_element = next(iter(training_logs))
            plot_line(
                plot_name=first_element['__source__'],
                line_name=f"Trial {index}",
                new_x_values=training_logs['total_number_of_episodes'],
                new_y_values=training_logs['per_episode_reward'],
            )
