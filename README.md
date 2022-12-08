# Setup

Note: only runs on Linux do to multiprocessing code. 
```sh
# install mujoco
commands/project/linux_install_mujoco
# setup python path and mujoco env vars
. commands/manual_init.sh
# install packages
pip install -e .
```

# Commands

Single run examples

```sh
python ./main/0_main.py
python ./main/0_main.py -- @CARTPOLE
python ./main/0_main.py -- @LUNAR_LANDER
python ./main/0_main.py -- @CHEETAH
python ./main/0_main.py -- @CARTPOLE attack_method:sign
python ./main/0_main.py -- @CARTPOLE attack_method:act
python ./main/0_main.py -- @CARTPOLE attack_method:noise
python ./main/0_main.py -- @CARTPOLE training:episode_count:21_000
python ./main/0_main.py -- @CARTPOLE number_of_processes:10 number_of_malicious_processes:3 expected_number_of_malicious_processes:3
```

Hyperparameter tuning

```sh
python ./main/0_hyp_tuning.py &>hyp_opt/EXPERIMENT_NAME.log
python ./main/0_hyp_tuning.py -- tuning:number_of_trials:300 &>hyp_opt/EXPERIMENT_NAME.log

# process the log files into usable data
commands/project/convert_hyper_opt_log hyp_opt/EXPERIMENT_NAME.log
# creates:
#     hyp_opt/EXPERIMENT_NAME.json
#     hyp_opt/EXPERIMENT_NAME.curves.json
#     figures/hyp_opt/EXPERIMENT_NAME.html
```

Comparisions

```sh
# will create a lot of files under logs/comparisons
python ./main/0_comparison.py 
# this will create a few html files under figures/comparisons/
python ./main/create_comparison_plots.py logs/comparisons/*.log
```