import logging
import sys
import os
import numpy as np

import optuna
from super_map import LazyDict

import train_a3c
from main.config import config, env_config

# dev_null = open(os.devnull, 'w')
# sys.stderr = f

def args_from_config():
    args = LazyDict()
    args.processes          =  16
    args.env                = "BreakoutNoFrameskip-v4"
    args.seed               =  0 # "Random seed [0, 2 ** 31)
    args.outdir             = "results" # ("Directory path to save output files. If it does not exist, it will be created."),)
    args.t_max              =  5
    args.beta               =  1e-2
    args.profile            = False
    args.steps              =  8 * 10**7
    args.max_frames         =  30 * 60 * 60, # 30 minutes with 60 fps help="Maximum number of frames for each episode.",
    args.lr                 =  7e-4
    args.eval_interval      =  250000
    args.eval_n_steps       =  125000
    args.demo               =  False
    args.load_pretrained    =  False
    args.pretrained_type    = "best" # choices=["best", "final"]
    args.load               =  ""
    args.log_level          =  20 # "Logging level. 10:DEBUG, 20:INFO etc."
    args.render             =  False # Render env states in a GUI window."
    args.monitor            =  False # Monitor env. Videos and additional information are saved as output files.
    args.permaban_threshold =  1
    args.malicious          =  0
    args.mal_type           = 'sign'
    args.rew_scale          =  1.0
    args.hidden_size        =  64
    args.activation         =  1
    
    # override with config
    args.seed      = config.random_seeds[0]
    args.processes = config.number_of_processes
    args.malicious = config.number_of_malicious_processes
    args.mal_type  = config.attack_method
    args.env                = env_config.env_name
    args.steps              = env_config.number_of_timesteps
    args.lr                 = env_config.learning_rate
    args.beta               = env_config.beta
    args.t_max              = env_config.t_max
    args.activation         = env_config.activation
    args.hidden_size        = env_config.hidden_size
    args.permaban_threshold = env_config.permaban_threshold
    
    return args

def objective(trial):
    args = args_from_config()
    
    # 
    # setup parameter options
    # 
    fl_high = -1
    fl_low = -8
    
    # descretize the learning rate options
    lr_base = trial.suggest_int("lr_b", 1, 5)
    lr_base = lr_base + (lr_base-1)
    lr_exp = trial.suggest_int("lr_e", fl_low, fl_high)
    args.lr = lr_base*(10**lr_exp)

    beta_base = trial.suggest_int("beta_b", 1, 5)
    beta_base = beta_base + (beta_base-1)
    beta_exp = trial.suggest_int("beta_e", fl_low, fl_high)
    args.beta = beta_base*(10**beta_exp)

    tmax_categories = [3, 5, 10, 20, 30, 50]
    tmax_cat = trial.suggest_int("tmax", 0, len(tmax_categories)-1)
    args.t_max = tmax_categories[tmax_cat]

    args.activation = trial.suggest_categorical("activ", [0, 1, 2])

    hid_categores = [16, 32, 64, 128]
    hid_cat = trial.suggest_int("hid", 0, len(hid_categores)-1)
    args.hidden_size = hid_categores[hid_cat]
    
    # 
    # run
    # 
    fitness_value = np.round(train_a3c.train_a3c(args), 2)
    return fitness_value


if __name__ == "__main__":
    import torch
    torch.multiprocessing.freeze_support()
    
    optuna.logging.disable_default_handler()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(optuna.logging.create_default_formatter())
    optuna.logging.get_logger("optuna").addHandler(sh)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(multivariate=True))
    study.optimize(objective, gc_after_trial=True)