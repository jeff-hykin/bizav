import argparse
import os

# Prevent numpy from using multiple threads
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np  # NOQA:E402
from torch import nn  # NOQA:E402

import pfrl  # NOQA:E402
from pfrl import experiments, utils  # NOQA:E402
from pfrl.agents import a3c  # NOQA:E402
from pfrl.optimizers import SharedRMSpropEpsInsideSqrt  # NOQA:E402
from pfrl.policies import SoftmaxCategoricalHead  # NOQA:E402
from pfrl.wrappers import atari_wrappers  # NOQA:E402

import logging
import torch
import gym

# 
# make warnings print a full stack trace
# 
import traceback
import warnings
import sys
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
warnings.showwarning = warn_with_traceback
warnings.simplefilter("always")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=16)
    parser.add_argument("--env", type=str, default="BreakoutNoFrameskip-v4")
    parser.add_argument("--seed", type=int, default=0, help="Random seed [0, 2 ** 31)")
    parser.add_argument(
        "--outdir",
        type=str,
        default="results",
        help=(
            "Directory path to save output files."
            " If it does not exist, it will be created."
        ),
    )
    parser.add_argument("--t-max", type=int, default=5)
    parser.add_argument("--beta", type=float, default=1e-2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--steps", type=int, default=8 * 10**7)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30 * 60 * 60,  # 30 minutes with 60 fps
        help="Maximum number of frames for each episode.",
    )
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--eval-interval", type=int, default=250000)
    parser.add_argument("--eval-n-steps", type=int, default=125000)
    parser.add_argument("--demo", action="store_true", default=False)
    parser.add_argument("--load-pretrained", action="store_true", default=False)
    parser.add_argument(
        "--pretrained-type", type=str, default="best", choices=["best", "final"]
    )
    parser.add_argument("--load", type=str, default="")
    parser.add_argument(
        "--log-level",
        type=int,
        default=20,
        help="Logging level. 10:DEBUG, 20:INFO etc.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render env states in a GUI window.",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help=(
            "Monitor env. Videos and additional information are saved as output files."
        ),
    )
    parser.add_argument("--ucb_disable", type=int, default=1)
    parser.add_argument("--malicious", type=float, default=0)
    parser.add_argument("--mal_type", type=str, default='sign')
    parser.add_argument("--rew_scale", type=float, default=1.0)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--activation", type=int, default=1)
    args = parser.parse_args()
    return args

one_above_max_seed = 2**31
def train_a3c(args):

    # Set a random seed used in PFRL.
    # If you use more than one processes, the results will be no longer
    # deterministic even with the same random seed.
    utils.set_random_seed(args.seed)

    # Set different random seeds for different subprocesses.
    # If seed=0 and processes=4, subprocess seeds are [0, 1, 2, 3].
    # If seed=1 and processes=4, subprocess seeds are [4, 5, 6, 7].
    process_seeds = np.arange(args.processes) + args.seed * args.processes
    assert process_seeds.max() < one_above_max_seed

    args.outdir = experiments.prepare_output_dir(args, args.outdir)
    # print("Output files are saved in {}".format(args.outdir))
    logging.basicConfig(level=args.log_level, filename=os.path.join(args.outdir, str(args.seed) + '.log'), force=True)

    def make_env(process_idx, test):
        # Use different random seeds for train and test envs
        process_seed = process_seeds[process_idx]
        env_seed = one_above_max_seed - 1 - process_seed if test else process_seed
        # env = atari_wrappers.wrap_deepmind(
        #     atari_wrappers.make_atari(args.env, max_frames=args.max_frames),
        #     episode_life=not test,
        #     clip_rewards=not test,
        # )
        env = gym.make(args.env)
        env = pfrl.wrappers.ScaleReward(env, args.rew_scale)
        if args.monitor:
            env = pfrl.wrappers.Monitor(
                env, args.outdir, mode="evaluation" if test else "training"
            )
        if args.render:
            env = pfrl.wrappers.Render(env)
        return env
    sample_env = make_env(0, False)
    obs_size = sample_env.observation_space.low.shape[0]

    if isinstance(sample_env.action_space, gym.spaces.Discrete):
        n_actions = sample_env.action_space.n
        def make_model(): return make_discrete_model(obs_size, n_actions, args.hidden_size, args.activation)
    elif isinstance(sample_env.action_space, gym.spaces.Box):
        n_actions = sample_env.action_space.low.size
        def make_model(): return make_continous_model(obs_size, n_actions, args.hidden_size, args.activation)
    else:
        raise NotImplementedError

    def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32)

    model = make_model()
    # SharedRMSprop is same as torch.optim.RMSprop except that it initializes
    # its state in __init__, allowing it to be moved to shared memory.
    opt = SharedRMSpropEpsInsideSqrt(model.parameters(), lr=args.lr, eps=1e-1, alpha=0.99)
    assert opt.state_dict()["state"], (
        "To share optimizer state across processes, the state must be"
        " initialized before training."
    )

    local_models = []
    for i in range(args.processes):
        local_models.append(make_model())

    agent = a3c.A3C(
        model,
        opt,
        t_max=args.t_max,
        gamma=0.99,
        beta=args.beta,
        phi=phi,
        max_grad_norm=40.0,
        malicious=args.malicious,
        mal_type=args.mal_type,
        local_models=local_models
    )

    if args.load or args.load_pretrained:
        # either load or load_pretrained must be false
        assert not args.load or not args.load_pretrained
        if args.load:
            agent.load(args.load)
        else:
            agent.load(
                utils.download_model("A3C", args.env, model_type=args.pretrained_type)[
                    0
                ]
            )

    if args.demo:
        env = make_env(0, True)
        eval_stats = experiments.eval_performance(
            env=env, agent=agent, n_steps=args.eval_n_steps, n_episodes=None
        )
        print(
            "n_steps: {} mean: {} median: {} stdev: {}".format(
                args.eval_n_steps,
                eval_stats["mean"],
                eval_stats["median"],
                eval_stats["stdev"],
            )
        )
    else:
        experiments.train_agent_async(
            agent=agent,
            outdir=args.outdir,
            processes=args.processes,
            make_env=make_env,
            profile=args.profile,
            steps=args.steps,
            eval_n_steps=args.eval_n_steps,
            eval_n_episodes=None,
            eval_interval=None,
            global_step_hooks=[],
            save_best_so_far_agent=True,
            num_agents_byz=args.malicious,
            step_before_disable=args.ucb_disable
        )
    mean_reward = get_results(os.path.join(args.outdir, str(args.seed) + '.log'), gym.spec(args.env).reward_threshold)
    return mean_reward

result_lookback_size = 50
def get_results(log_file, thresh):
    rewards = []
    with open(log_file) as fp:
        for line in fp:
            if 'Saved' in line: continue
            rewards.append(float(line[22:].split(';')[2].strip()))
    
    last_x_results = rewards[-result_lookback_size:]
    if not last_x_results:
        return 0
    last_eps = np.mean(last_x_results)
    if last_eps >= thresh: return last_eps + np.mean(rewards)
    return last_eps


def get_activation(activation):
    if(activation == 0): return nn.ReLU
    if(activation == 1): return nn.Tanh
    if(activation == 2): return nn.GELU


def make_discrete_model(obs_size, n_actions, hidden_size, activation):
    activation = get_activation(activation)
    return nn.Sequential(
        nn.Linear(obs_size, hidden_size),
        activation(),
        nn.Linear(hidden_size, hidden_size),
        activation(),
        nn.Linear(hidden_size, hidden_size),
        activation(),
        pfrl.nn.Branched(
            nn.Sequential(
                nn.Linear(hidden_size, n_actions),
                SoftmaxCategoricalHead(),
            ),
            nn.Linear(hidden_size, 1),
        ),
    )


def make_continous_model(obs_size, action_size, hidden_size, activation):
    activation = get_activation(activation)
    return torch.nn.Sequential(
        nn.Linear(obs_size, hidden_size),
        activation(),
        nn.Linear(hidden_size, hidden_size),
        activation(),
        nn.Linear(hidden_size, hidden_size),
        activation(),
        pfrl.nn.Branched(
            nn.Sequential(
                nn.Linear(hidden_size, action_size),
                pfrl.policies.GaussianHeadWithStateIndependentCovariance(
                    action_size=action_size,
                    var_type="diagonal",
                    var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
                    var_param_init=0,  # log std = 0 => std = 1
                )
            ),
            nn.Linear(hidden_size, 1)
        ),
    )


if __name__ == "__main__":
    args = parse_args()
    train_a3c(args)
