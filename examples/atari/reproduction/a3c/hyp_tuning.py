import logging
import sys
import os
import numpy as np

import optuna
from super_map import LazyDict

import train_a3c
from train_a3c import args_from_config
from main.config import config, env_config, info, args as cli_args
from blissful_basics import FS

# dev_null = open(os.devnull, 'w')
# sys.stderr = f

def phase1_tuning(trial):
    # 
    # modify the config
    # 
    hyper_modify_config(
        trial,
        hyper_options=config.tuning.phase_1,
        env_config=env_config,
    )
        
    # 
    # run
    # 
    args = args_from_config() 
    fitness_value = float(train_a3c.train_a3c(args, trial))
    return fitness_value

def phase2_tuning(trial):
    
    # 
    # modify the config
    # 
    hyper_modify_config(
        trial,
        hyper_options=config.tuning.phase_2,
        env_config=env_config,
    )
    
    # 
    # run
    # 
    args = args_from_config()
    fitness_value = float(train_a3c.train_a3c(args, trial))
    return fitness_value

def hyper_modify_config(trial, hyper_options, env_config):
    # categorical_options
    for each_key, each_set_of_possibilitites in hyper_options.get("categorical_options",{}).items():
        env_config[each_key] = trial.suggest_categorical(each_key, each_set_of_possibilitites)
    # sequential_options
    for each_key, each_sequence_of_possibilitites in hyper_options.get("sequential_options",{}).items():
        index_for_sequence = trial.suggest_int(each_key, 0, len(each_sequence_of_possibilitites)-1)
        env_config[each_key] = each_sequence_of_possibilitites[index_for_sequence]
    # sequential_exponential_options
    for each_key, each_info in hyper_options.get("sequential_exponential_options",{}).items():
        base_options     = each_info["base"]
        exponent_options = each_info["exponent"]
        index_for_base     = trial.suggest_int(each_key+"_base"    , 0, len(base_options)-1)
        index_for_exponent = trial.suggest_int(each_key+"_exponent", 0, len(exponent_options)-1)
        
        base = base_options[index_for_base]
        exponent = exponent_options[index_for_exponent]
        env_config[each_key] = base*(10**exponent)

def start_running_trials(objective, number_of_trials):
    try:
        study_name = cli_args[0]
    except Exception as error:
        raise Exception(f'''\n\nPlease pass a study name as the first CLI argument''')
    
    
    optuna.logging.disable_default_handler()
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(optuna.logging.create_default_formatter())
    optuna.logging.get_logger("optuna").addHandler(stream_handler)
    FS.ensure_is_folder(info.absolute_path_to.study_checkpoints)
    
    storage = optuna.storages.RDBStorage(
        f"sqlite:///{info.path_to.study_checkpoints}/{study_name}.db",
        heartbeat_interval=1,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(),
    )
    study = optuna.create_study(
        storage=storage,
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(multivariate=True),
        direction="maximize",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=number_of_trials, gc_after_trial=True)

    pruned_trials = study.get_trials(states=(optuna.trial.TrialState.PRUNED,))
    complete_trials = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # The line of the resumed trial's intermediate values begins with the restarted epoch.
    optuna.visualization.plot_intermediate_values(study).show()

if __name__ == "__main__":
    import torch
    torch.multiprocessing.freeze_support()
    
    start_running_trials(objective=phase1_tuning, number_of_trials=config.tuning.number_of_trials)