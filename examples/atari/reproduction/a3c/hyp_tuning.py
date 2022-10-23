import logging
import sys
import os
import numpy as np

import optuna
from super_map import LazyDict

import train_a3c
from train_a3c import args_from_config
from main.config import config, env_config

# dev_null = open(os.devnull, 'w')
# sys.stderr = f

def stage1_tuning(trial):
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
    fitness_value = float(train_a3c.train_a3c(args))
    return fitness_value

def stage2_tuning(trial):
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
    fitness_value = float(train_a3c.train_a3c(args))
    return fitness_value

def hyper_modify_config(trial, hyper_options, env_config):
    options = config.tuning.phase_1
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

if __name__ == "__main__":
    import torch
    torch.multiprocessing.freeze_support()
    
    optuna.logging.disable_default_handler()
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(optuna.logging.create_default_formatter())
    optuna.logging.get_logger("optuna").addHandler(stream_handler)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(multivariate=True))
    study.optimize(stage2_tuning, gc_after_trial=True)