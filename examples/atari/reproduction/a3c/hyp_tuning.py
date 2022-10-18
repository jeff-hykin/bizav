import logging
import sys
import os
import numpy as np

import optuna

import train_a3c


f = open(os.devnull, 'w')
sys.stderr = f


def objective(trial):
    args = train_a3c.parse_args()
    args.env = "LunarLander-v2"
    args.steps = 100000

    fl_high = -1
    fl_low = -8

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

    return np.round(train_a3c.train_a3c(args), 2)


optuna.logging.disable_default_handler()
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(optuna.logging.create_default_formatter())
optuna.logging.get_logger("optuna").addHandler(sh)

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(multivariate=True))
study.optimize(objective, gc_after_trial=True)
