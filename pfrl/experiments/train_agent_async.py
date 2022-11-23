import logging
import os
import signal
import subprocess
import sys
from random import random, sample, choices
from statistics import mean
import math
import json
import threading

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn
from super_map import LazyDict
from blissful_basics import singleton, max_indices, print, to_pure, shuffled

from pfrl.experiments.evaluator import AsyncEvaluator
from pfrl.utils import async_, random_seed

from main import utils
from main.config import config, env_config, info
from main.utils import trend_calculate

# 
# constants
# 
debug = False
check_rate = config.number_of_processes
use_softmax_defense  = config.defense_method == 'softmax'
use_permaban_defense = config.defense_method == 'permaban'
use_no_defense       = not (use_softmax_defense or use_permaban_defense)

# 
# globals
# 
successfully_filtered_sum       = None
successfully_filtered_increment = None
prev_total_number_of_episodes   = None
number_of_timesteps             = None
number_of_episodes              = None
number_of_updates               = None
filtered_count                  = None
episode_reward_trend            = None
process_gradient_sum            = None
process_q_value                 = None
process_temp_banned_count       = None
process_permabanned             = None
process_median_episode_rewards  = None
process_temp_ban                = None
process_is_central_agent        = None
process_is_malicious            = None
process_latest_episode_reward   = None
process_latest_eval_score       = None
process_accumulated_distance    = None
processes                       = None
process_gradients               = None

class Process:
    def __init__(self, process_index):
        self.index = process_index
    
    @property
    def is_banned(self):
        if use_no_defense:
            return False
        else:
            return self.is_permabanned or self.is_temp_banned
    
    # 
    # permaban
    # 
    @property
    def accumulated_distance(self): return process_accumulated_distance[self.index]
    @accumulated_distance.setter
    def accumulated_distance(self, value): process_accumulated_distance[self.index] = value
    
    # 
    # permaban
    # 
    @property
    def is_permabanned(self): return process_permabanned[self.index]
    @is_permabanned.setter
    def is_permabanned(self, value): process_permabanned[self.index] = value
    
    
    # 
    # temp ban
    # 
    @property
    def is_temp_banned(self):
        if process_temp_ban[self.index]:
            return True
        
        return False
    
    # 
    # temp ban
    # 
    @property
    def is_malicious(self):
        if process_is_malicious[self.index]:
            return True
        
        return False
    
    # 
    # q value
    # 
    @property
    def q_value(self): return process_q_value[self.index]
    @q_value.setter
    def q_value(self, value): process_q_value[self.index] = value
    
    # 
    # visits
    # 
    @property
    def temp_ban_count(self): return process_temp_banned_count[self.index]
    @temp_ban_count.setter
    def temp_ban_count(self, value): process_temp_banned_count[self.index] = value
    
    # 
    # median rewards (training)
    # 
    @property
    def median_episode_reward(self): return process_median_episode_rewards[self.index]
    @median_episode_reward.setter
    def median_episode_reward(self, value): process_median_episode_rewards[self.index] = value
    
    # 
    # rewards training
    # 
    @property
    def latest_episode_reward(self): return process_latest_episode_reward[self.index]
    @latest_episode_reward.setter
    def latest_episode_reward(self, value): process_latest_episode_reward[self.index] = value
    
    # 
    # rewards training
    # 
    @property
    def latest_eval_score(self): return process_latest_eval_score[self.index]
    @latest_eval_score.setter
    def latest_eval_score(self, value): process_latest_eval_score[self.index] = value
    
    # 
    # is_central_agent
    # 
    @property
    def is_central_agent(self): return process_is_central_agent == self.index

def reset_globals():
    global process_is_central_agent, prev_total_number_of_episodes, number_of_timesteps, number_of_episodes, number_of_updates,  filtered_count,  process_q_value,  process_permabanned, episode_reward_trend, process_median_episode_rewards, process_temp_ban, process_temp_banned_count, processes, process_is_malicious, process_accumulated_distance, process_gradient_sum, process_gradients, process_latest_episode_reward, process_latest_eval_score, successfully_filtered_sum, successfully_filtered_increment
    episode_reward_trend = []
    process_is_central_agent = sample(list(range(config.number_of_processes)), k=1)[0]
    prev_total_number_of_episodes = -1
    
    
    # 
    # init shared values
    # 
    number_of_timesteps             = mp.Value("l", 0)
    number_of_episodes              = mp.Value("l", 0)
    number_of_updates               = mp.Value("l", 0) # Number of total rollouts completed
    filtered_count                  = mp.Value("l", 0) # Number of permanently filtered agents
    successfully_filtered_sum       = mp.Value("d", 0)
    successfully_filtered_increment = mp.Value("l", 0)

    process_q_value                  = mp.Array("d", config.number_of_processes) # Q-values
    process_temp_banned_count        = mp.Array("d", config.number_of_processes) # number of process_temp_banned_count
    process_permabanned              = mp.Array("l", config.number_of_processes) # Permanently filtered agent memory
    process_median_episode_rewards   = mp.Array("d", config.number_of_processes)
    process_temp_ban                 = mp.Array("l", config.number_of_processes) 
    process_is_malicious             = mp.Array("l", config.number_of_processes) 
    process_accumulated_distance     = mp.Array("d", config.number_of_processes) 
    process_gradient_sum             = mp.Array("d", config.number_of_processes) 
    process_latest_episode_reward    = mp.Array("d", config.number_of_processes) 
    process_latest_eval_score        = mp.Array("d", config.number_of_processes) 
    process_gradients                = mp.Array("d", config.number_of_processes * config.env_config.gradient_size)
    for process_index in range(config.number_of_processes):
        process_q_value[process_index]               = 0
        process_permabanned[process_index]           = 0
        process_is_malicious[process_index]          = 0
        process_accumulated_distance[process_index]  = 0
        process_gradient_sum[process_index]          = 0
        process_latest_episode_reward[process_index] = 0
        process_latest_eval_score[process_index]     = 0
        process_temp_banned_count[process_index]     = 1 # ASK: avoids a division by 0, but treats all processes equal
    
    malicious_indices = shuffled(list(range(config.number_of_processes)))[0:config.number_of_malicious_processes]
    for each_index in malicious_indices:
        process_is_malicious[each_index] = 1
    
    processes = tuple(Process(each_index) for each_index in range(config.number_of_processes))

def kill_all():
    if os.name == "nt":
        # windows
        # taskkill with /T kill all the subprocess
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(os.getpid())])
    else:
        pgid = os.getpgrp()
        os.killpg(pgid, signal.SIGTERM)
        sys.exit(1)

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    import torch
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).T
    dist = xx + yy
    dist.addmm_(1, -2, x, y.T)
    dist[dist < 0] = 0
    dist = dist.sqrt()
    return dist

@print.indent.function
def inner_training_loop(
    process_idx,
    env,
    agent,
    max_number_of_episodes,
    outdir,
    number_of_episodes,
    stop_event,
    exception_event,
    process_median_episode_rewards,
    max_episode_len=None,
    evaluator=None,
    eval_env=None,
    logger=None,
    global_step_hooks=[],
    individual_updates_ready_barrier=None,
    individual_updates_contributed_barrier=None,
    process_permabanned=None,
    expected_number_of_malicious_processes=0
):
    global episode_reward_trend
    max_number_of_episodes = config.training.episode_count # override arg (could use cleaning up)
    config.verbose and print("[starting inner_training_loop()]", flush=True)
    logger = logger or logging.getLogger(__name__)
    process = Process(process_idx)

    if eval_env is None:
        eval_env = env

    def save_model():
        if process.is_central_agent:
            # Save the current model before being killed
            dirname = os.path.join(outdir, "{}_except".format(number_of_episodes.value))
            agent.save(dirname)
            logger.info("Saved the current model to %s", dirname)

    try:
        observation     = env.reset()
        episode_reward  = 0
        episode_len     = 0
        number_of_timesteps_for_this_episode = 0

        while True:
            # a_t
            action = agent.act(observation)
            # o_{t+1}, r_{t+1}
            observation, reward, done, info = env.step(action)
            number_of_timesteps_for_this_episode += 1
            episode_reward += reward
            episode_len += 1
            reset = episode_len == max_episode_len or info.get("needs_reset", False)
            agent.observe(observation, reward, done, reset)
            
            
            debug and print(f'''agent{process.index}.updated = {agent.updated}''')
            if agent.updated:
                debug and print(f'''agent.gradient.shape = {agent.gradient.shape}''')
                process_gradient_sum[process.index] = agent.gradient_sum
                start = process.index * config.env_config.gradient_size 
                end   = (process.index+1) * config.env_config.gradient_size 
                process_gradients[start:end] = agent.gradient
                number_of_non_zero = sum(1 for each in process_gradients if each != 0)
                debug and print(f'''non zero process_gradients = {number_of_non_zero/config.env_config.gradient_size}''')
                try:
                    individual_updates_ready_barrier.wait()  # Wait for all agents to complete rollout, then run when_all_processes_are_updated()
                except Exception as error:
                    print(f"exited at individual_updates_ready_barrier.wait(): {process.index}, error = {error}")
                    exit()
                if not process.is_banned:
                    debug and print(f'''agent{process.index}.add_update()''')
                    # include agent's gradient in global model
                    agent.add_update()
                try:
                    individual_updates_contributed_barrier.wait()   # Wait for all agents to contribute their gradients to global model, the sync_updates() to step it's optimizer
                except threading.BrokenBarrierError as error:
                    print(f"Barrier broken at individual_updates_contributed_barrier.wait(): {process.index}, error = {error}")
                except Exception as error:
                    print(f"exited at individual_updates_contributed_barrier.wait(): {process.index}, error = {error}")
                    raise error
                agent.after_update()    # Each agent will download the global model after the optimizer steps

            if done or reset:# Get and increment the global number_of_episodes
                with number_of_episodes.get_lock():
                    process.latest_episode_reward = episode_reward
                    process.median_episode_reward = agent.median_reward_per_episode
                    number_of_episodes.value += 1
                    number_of_timesteps.value += number_of_timesteps_for_this_episode
                
                # reset
                number_of_timesteps_for_this_episode = 0
                
                for hook in global_step_hooks:
                    hook(env, agent, number_of_episodes.value)

                metric_line = str(number_of_episodes.value)+'; '+str(process.index)+'; '+str(episode_reward)+'; '
                stats = agent.get_statistics()
                for item in stats:
                    metric_line += str(item) + '; '
                if stats[0] != -1:
                    logger.info(metric_line[:-2])

                # Evaluate the current agent
                if evaluator is not None:
                    eval_score = evaluator.evaluate_if_necessary(
                        # eval is triggered based on t (timesteps), but its flexible, so we trigger it based on episodes instead
                        t=number_of_episodes.value, # ASK
                        episodes=number_of_episodes.value,
                        env=eval_env,
                        agent=agent,
                    )
                    if eval_score != None:
                        process.latest_eval_score = eval_score
                        import json
                        # ASK: does the agent malicious/non-malicious matter since the global update was just done
                        print(json.dumps(dict(eval_score=eval_score, number_of_episodes=number_of_episodes.value, is_malicious=process.is_malicious, agent=process.index)))
                
                proportional_number_of_timesteps = number_of_timesteps.value / config.number_of_processes
                if number_of_episodes.value >= max_number_of_episodes or stop_event.is_set():
                    break

                # Start a new episode
                episode_reward = 0
                episode_len = 0
                observation = env.reset()

            if process.is_central_agent and exception_event.is_set():
                logger.exception("An exception detected, exiting")
                save_model()
                kill_all()
        
        debug and print("aborting since agent exited while loop")
        individual_updates_contributed_barrier.abort()
        individual_updates_ready_barrier.abort()

    except (Exception, KeyboardInterrupt):
        save_model()
        raise

    if number_of_episodes.value == max_number_of_episodes:
        # Save the final model
        dirname = os.path.join(outdir, "{}_finish".format(max_number_of_episodes))
        agent.save(dirname)
        logger.info("Saved the final agent to %s", dirname)

def middle_training_function(
    outdir,
    make_env,
    profile=False,
    steps=8 * 10**7,
    eval_interval=10**6,
    eval_n_steps=None,
    eval_n_episodes=10,
    eval_success_threshold=0.0,
    max_episode_len=None,
    step_offset=0,
    agent=None,
    make_agent=None,
    global_step_hooks=[],
    evaluation_hooks=(),
    save_best_so_far_agent=True,
    use_tensorboard=False,
    logger=None,
    random_seeds=None,
    stop_event=None,
    exception_event=None,
    use_shared_memory=True,
    permaban_threshold=1,
    trial=None,
):
    """Train agent asynchronously using multiprocessing.

    Either `agent` or `make_agent` must be specified.

    Args:
        outdir (str): Path to the directory to output things.
        make_env (callable): (process_idx, test) -> Environment.
        profile (bool): Profile if set True.
        steps (int): Number of global time steps for training.
        eval_interval (int): Interval of evaluation. If set to None, the agent
            will not be evaluated at all.
        eval_n_steps (int): Number of eval timesteps at each eval phase
        eval_n_episodes (int): Number of eval episodes at each eval phase
        eval_success_threshold (float): r-threshold above which grasp succeeds
        max_episode_len (int): Maximum episode length.
        step_offset (int): Time step from which training starts.
        agent (Agent): Agent to train.
        make_agent (callable): (process_idx) -> Agent
        global_step_hooks (list): List of callable objects that accepts
            (env, agent, step) as arguments. They are called every global
            step. See pfrl.experiments.hooks.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation,
            if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
        logger (logging.Logger): Logger used in this function.
        random_seeds (array-like of ints or None): Random seeds for processes.
            If set to None, [0, 1, ..., processes-1] are used.
        stop_event (multiprocessing.Event or None): Event to stop training.
            If set to None, a new Event object is created and used internally.
        exception_event (multiprocessing.Event or None): Event that indicates
            other thread raised an excpetion. The train will be terminated and
            the current agent will be saved.
            If set to None, a new Event object is created and used internally.
        use_shared_memory (bool): Share memory amongst asynchronous agents.

    Returns:
        Trained agent.
    """
    global process_is_central_agent, prev_total_number_of_episodes, number_of_timesteps, number_of_episodes, number_of_updates,  filtered_count,  process_q_value,  process_temp_banned_count,  process_permabanned, episode_reward_trend, process_median_episode_rewards, episode_reward_trend
    
    config.verbose and print("[starting middle_training_function()]\n")
    logger = logger or logging.getLogger(__name__)

    for hook in evaluation_hooks:
        if not hook.support_middle_training_function:
            raise ValueError("{} does not support middle_training_function().".format(hook))
    
    # Prevent numpy from using multiple threads
    os.environ["OMP_NUM_THREADS"] = "1"
    
    reset_globals()
    
    # 
    # create UCB manager
    # 
    @singleton
    class ucb:
        raw_value_per_process = process_q_value
        
        @print.indent.function
        def reward_func(self, all_grads):
            if use_permaban_defense:
                probably_good_gradients = []
                for process, each_gradient in zip(processes, all_grads):
                    if not process.is_banned:
                        probably_good_gradients.append(each_gradient)
                
                agent_gradients = torch.tensor(np.vstack(probably_good_gradients))
                average_distance = euclidean_dist(agent_gradients, agent_gradients).mean()
                flipped_value = -average_distance
                ucb_reward = config.env_config.variance_scaling_factor * flipped_value
                return [ucb_reward]
            
            if use_softmax_defense:
                reward_for_each_temp_banned_index = []
                for each_process_being_reviewed in processes:
                    all_other_gradients = []
                    indices_of_others = []
                    for process, each_gradient in zip(processes, all_grads):
                        # TODO: consider this alternative (could eliminate one for loop)
                        # if process_index in previously_temp_banned_indices:
                        #     continue
                        if process.index != each_process_being_reviewed.index:
                            all_other_gradients.append(each_gradient)
                            indices_of_others.append(process.index)
                    
                    all_other_gradients = torch.tensor(np.vstack(all_other_gradients))
                    # debug and print(f'''all_other_gradients.shape = {all_other_gradients.shape}''')
                    # for process_index, each_gradient in zip(indices_of_others, all_other_gradients):
                        # debug and print(f'''    agent{process_index} gradient sum = {each_gradient.sum()}''')
                    process_distances = euclidean_dist(all_other_gradients, torch.vstack([ torch.tensor(all_grads[each_process_being_reviewed.index])]))
                    # debug and print(f'''process_distances = {process_distances}''')
                    mean_process_distance = process_distances.mean()
                    # debug and print(f'''mean_process_distance = {mean_process_distance}''')
                    debug and print(f'''math.log(mean_process_distance) = {math.log(mean_process_distance)}, is_malicious: {each_process_being_reviewed.is_malicious}''')
                    each_process_being_reviewed.accumulated_distance += mean_process_distance
                    flipped_value = -mean_process_distance
                    ucb_reward = config.env_config.variance_scaling_factor * flipped_value
                    reward_for_each_temp_banned_index.append(ucb_reward)
                        
                return reward_for_each_temp_banned_index
                
        def update_step(self, ucb_reward):
            if use_softmax_defense:
                pass
            else:
                for each_process in processes:
                    # update temp_banned processes
                    if each_process.is_temp_banned:
                        old_q_value      = each_process.q_value
                        number_of_visits = each_process.temp_ban_count
                        change_in_value  = (ucb_reward.pop(0) - old_q_value) / number_of_visits
                        # apply the new value
                        each_process.q_value += change_in_value
                    
                    if each_process.is_permabanned:
                        each_process.q_value = -math.inf
                
        def value_of_each_process(self):
            np_process_is_malicious      = mp_to_numpy(process_is_malicious)
            np_process_temp_ban          = mp_to_numpy(process_temp_ban)
            np_process_temp_banned_count = mp_to_numpy(process_temp_banned_count)
            np_value_per_process         = mp_to_numpy(ucb.raw_value_per_process)
            # TODO: clean this up
            # successful = sum(np_process_is_malicious * process_temp_ban)
            # config.verbose and print(json.dumps(dict(
            #     step=number_of_updates.value,
            #     successfully_filtered=successful,
            #     filter_choice=process_temp_ban,
            #     process_temp_banned_count=list(np.round(np_process_temp_banned_count, 2)),
            #     q_vals=list(np.round(np_value_per_process, 3)),
            # )))
            
            if use_permaban_defense:
                # Get the true UCB t value
                ucb_timesteps = np.sum(np_process_temp_banned_count) - (env_config.permaban_threshold+1) * filtered_count.value
                # Compute UCB policy values (Q-value + uncertainty)
                return np_value_per_process + np.sqrt((np.log(ucb_timesteps)) / np_process_temp_banned_count)
            
            if use_softmax_defense:
                if sum(process_temp_ban) == 0:
                    previously_chosen_indices = tuple()
                else:
                    previously_chosen_indices = max_indices(process_temp_ban)
                
                number_of_banned_items = (config.expected_number_of_malicious_processes * number_of_updates.value)
                ucb_timesteps = np.sum(np_process_temp_banned_count) - number_of_banned_items
                # Compute UCB policy values (Q-value + uncertainty)
                return np_value_per_process + np.sqrt((np.log(ucb_timesteps)) / np_process_temp_banned_count)
                
        def choose_action(self):
            output = None
            if use_permaban_defense:
                # Initial selection (process_temp_banned_count 0) #TODO properly select at random
                np_process_temp_banned_count = mp_to_numpy(process_temp_banned_count)
                if np.min(np_process_temp_banned_count) < 1:
                    output = [ np.argmin(np_process_temp_banned_count) ]
                else:
                    output = [ np.argmax(ucb.value_of_each_process()) ]
            
            elif use_softmax_defense:
                debug and print(f'''process_accumulated_distance = {list(process_accumulated_distance)}''')
                weights = list(to_pure(process_accumulated_distance))
                debug and print(f'''raw weights = {weights}''')
                process_indices = list(range(config.number_of_processes))
                
                minimum = min(weights)
                weights = [ (each-minimum+1)**5 for each in weights ]
                debug and print(f'''exaggerated weights = {weights}''')
                
                # issue when all weights are 0
                if any(each == 0 for each in weights):
                    debug and print("picking random indicies")
                    return shuffled(process_indices)[0:config.number_of_malicious_processes]
                
                choices = set()
                import random
                while len(choices) < config.expected_number_of_malicious_processes:
                    remaining_indices_and_weights = [ (index, weight) for index, weight in zip(process_indices, weights) if index not in choices ]
                    remaining_indicies = [ index for index, weight in remaining_indices_and_weights ]
                    remaining_weights  = [ weight for index, weight in remaining_indices_and_weights ]
                    random_index = random.choices(
                        remaining_indicies,
                        weights=remaining_weights,
                        k=1,
                    )[0]
                    choices.add(random_index)
                
                # who to ban this round
                output = list(choices)
                debug and print(f'''output = {output}''')
            
                np_process_is_malicious      = mp_to_numpy(process_is_malicious)
                np_process_temp_ban          = mp_to_numpy(process_temp_ban)
                np_process_temp_banned_count = mp_to_numpy(process_temp_banned_count)
                np_value_per_process         = mp_to_numpy(ucb.raw_value_per_process)
                successful = sum(np_process_is_malicious * process_temp_ban)
                
                successfully_filtered_sum.value += successful
                successfully_filtered_increment.value += 1
                
                from statistics import mean
                config.verbose and print(json.dumps(dict(
                    step=number_of_updates.value,
                    successfully_filtered_avg=round(successfully_filtered_sum.value/successfully_filtered_increment.value, 2),
                    successfully_filtered=successful,
                    malicious=list(process_is_malicious),
                    weights=[ round(each, 3) for each in weights ],
                    filter_choice=process_temp_ban,
                    process_temp_banned_count=list(np.round(np_process_temp_banned_count, 2)),
                    q_vals=list(np.round(np_value_per_process, 3)),
                )))
            
            return output
                
        @property
        def gradient_of_agents(self):
            all_grads = []
            for process in processes:
                my_grad = []
                for param in agent.local_models[process.index].parameters():
                    if param.grad is not None:
                        grad_np = param.grad.detach().clone().numpy().flatten()
                    else:
                        grad_np = np.zeros(param.size(), dtype=np.float).flatten()
                    for j in range(len(grad_np)):
                        my_grad.append(grad_np[j])
                all_grads.append(np.asarray(my_grad))
            
            debug and print("all_grads")
            with print.indent:
                for each in all_grads:
                    debug and print(f'''each.sum() = {each.sum()}''')
            return np.vstack(all_grads)
        
    def early_stopping_check():
        global process_is_central_agent, prev_total_number_of_episodes, number_of_timesteps, number_of_episodes, number_of_updates, filtered_count, process_q_value, process_temp_banned_count, process_permabanned, episode_reward_trend, process_median_episode_rewards, episode_reward_trend
        
        # 
        # early stopping check
        # 
        total_number_of_episodes = number_of_episodes.value
        if total_number_of_episodes > config.early_stopping.min_number_of_episodes:
            
            # limiter
            if total_number_of_episodes >= prev_total_number_of_episodes + check_rate:
                prev_total_number_of_episodes = total_number_of_episodes
                relevent_training_rewards = []
                relevent_eval_scores = []
                # reset the rewards of filtered-out agents
                for each_process in processes:
                    if each_process.is_permabanned:
                        each_process.median_episode_reward = 0
                    elif each_process.is_banned:
                        continue
                    else:
                        relevent_training_rewards.append(each_process.latest_episode_reward)
                        relevent_eval_scores.append(each_process.latest_eval_score)
                from statistics import mean as average
                latest_episode_reward = average(relevent_training_rewards)
                latest_eval_score = average(relevent_eval_scores)
                episode_reward_trend.append(latest_episode_reward)
                episode_reward_trend = episode_reward_trend[-config.value_trend_lookback_size:]
                episode_reward_trend_value = trend_calculate(episode_reward_trend) / check_rate
                biggest_recent_change = math.nan
                
                # check optuna stopper
                if trial:
                    import optuna
                    trial.report(latest_episode_reward, total_number_of_episodes)
                    if trial.should_prune():
                        stop_event.set()
                        raise optuna.TrialPruned()
                
                if len(episode_reward_trend) >= config.value_trend_lookback_size:
                    absolute_changes = [ abs(each) for each in utils.sequential_value_changes(episode_reward_trend)  ]
                    biggest_recent_change = max(absolute_changes)
                    if trial and biggest_recent_change < config.early_stopping.lowerbound_for_max_recent_change:
                        print(f"Hit early stopping because biggest_recent_change: {biggest_recent_change} < {config.early_stopping.lowerbound_for_max_recent_change}")
                        stop_event.set()
                        raise optuna.TrialPruned()
                
                print(json.dumps(dict(
                    number_of_episodes=total_number_of_episodes,
                    number_of_timesteps=number_of_timesteps.value,
                    latest_eval_score=latest_eval_score,
                    latest_episode_reward=round(latest_episode_reward, 2),
                    episode_reward_trend_value=episode_reward_trend_value,
                    biggest_recent_change=biggest_recent_change,
                )))
                
                # only do early stopping if tuning hyperparameters
                if trial:
                    for each_step, each_min_value in config.early_stopping.thresholds.items():
                        # if meets the increment-based threshold
                        if total_number_of_episodes > each_step:
                            # enforce that it return the minimum 
                            if latest_episode_reward < each_min_value:
                                print(f"Hit early stopping because latest_episode_reward: {latest_episode_reward} < {each_min_value}")
                                stop_event.set()
                                raise optuna.TrialPruned()
    
    @print.indent.function
    def when_all_processes_are_updated():
        global process_is_central_agent, prev_total_number_of_episodes, number_of_timesteps, number_of_episodes, number_of_updates,  filtered_count,  process_q_value,  process_temp_banned_count,  process_permabanned, episode_reward_trend, process_median_episode_rewards, episode_reward_trend, process_gradients
        try:
            import numpy
            debug and print("started individual_updates_ready_barrier()")
            debug and print(f'''process_gradient_sum = {list(process_gradient_sum)}''')
            gradients_np = numpy.asarray(list(process_gradients))
            debug and print(f'''gradients_np = {gradients_np}''')
            all_grads = gradients_np.reshape((config.number_of_processes, config.env_config.gradient_size))
            debug and print(f'''all_grads.sum(axis=1) = {all_grads.sum(axis=1).tolist()}''')
            debug and print("starting early_stopping_check()")
            early_stopping_check()
            debug and print("finished early_stopping_check()")
            
            all_malicious_actors_found = filtered_count.value == config.expected_number_of_malicious_processes
            if all_malicious_actors_found:
                debug and print("finished individual_updates_ready_barrier()")
                return
            
            # Update values
            debug and print("number_of_updates.value", number_of_updates.value)
            if number_of_updates.value != 0:
                # Compute gradient mean
                debug and print("starting reward_func()")
                ucb_reward = ucb.reward_func(ucb.gradient_of_agents) # FIXME: ucb.reward_func(all_grads) is more accurate... but doesnt work as well for some reason
                
                # Update Q-values
                debug and print("starting update_step()")
                ucb.update_step(ucb_reward)
                
                if use_permaban_defense:
                    # 
                    # Permanently disable an agent
                    # 
                    prev_amount_filtered = filtered_count.value
                    for process_index, ban_count in enumerate(process_temp_banned_count):
                        if ban_count >= env_config.permaban_threshold:
                            process_permabanned[process_index] = 1
                    filtered_count.value = sum(process_permabanned)
                    
                    # 
                    # if someone was just recently permabaned
                    # 
                    if prev_amount_filtered < filtered_count.value:
                        # Reset non-disabled process_temp_banned_count/Q-values
                        for process_index, ban_count in enumerate(process_temp_banned_count):
                            if ban_count < env_config.permaban_threshold:
                                process_temp_banned_count[process_index] = 0
                                ucb.raw_value_per_process[process_index] = 0
            
            # Select next action
            debug and print("starting choose_action()")
            who_to_ban = ucb.choose_action()
            debug and print(f"finished choose_action(), who_to_ban={who_to_ban}")
            if who_to_ban:
                for each_process in processes:
                    if each_process.index in who_to_ban:
                        each_process.temp_ban_count += 1
                        process_temp_ban[each_process.index] = 1 # aka True
                    else:
                        process_temp_ban[each_process.index] = 0
            number_of_updates.value += 1
            
            debug and print("finished individual_updates_ready_barrier()")
        
        except Exception as error:
            print(f'''error = {error}''')
            raise error
        
    individual_updates_ready_barrier = mp.Barrier(config.number_of_processes, when_all_processes_are_updated)

    def sync_updates():
        try:
            if filtered_count.value != config.expected_number_of_malicious_processes:
                num_updates = config.number_of_processes - (filtered_count.value + 1)
            else:
                num_updates = config.number_of_processes - filtered_count.value
            agent.average_updates(num_updates)
            agent.optimizer.step()
        except Exception as error:
            print(f'''sync_updates: error = {error}''')
    individual_updates_contributed_barrier = mp.Barrier(config.number_of_processes, sync_updates)

    if stop_event is None:
        stop_event = mp.Event()

    if exception_event is None:
        exception_event = mp.Event()

    if use_shared_memory:
        if agent is None:
            assert make_agent is not None
            agent = make_agent(0)

        # Move model and optimizer states in shared memory
        for attr in agent.shared_attributes:
            attr_value = getattr(agent, attr)
            if isinstance(attr_value, nn.Module):
                for k, v in attr_value.state_dict().items():
                    v.share_memory_()
            elif isinstance(attr_value, torch.optim.Optimizer):
                for param, state in attr_value.state_dict()["state"].items():
                    assert isinstance(state, dict)
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            v.share_memory_()
            elif isinstance(attr_value, list):  # Local models
                for item in attr_value:
                    if isinstance(item, nn.Module):
                        for k, v in item.state_dict().items():
                            v.share_memory_()

    if eval_interval is None:
        evaluator = None
    else:
        evaluator = AsyncEvaluator(
            n_steps=eval_n_steps,
            n_episodes=eval_n_episodes,  # eval_n_episodes = config.evaluation.number_of_epsiodes_during_eval,
            eval_interval=eval_interval, # eval_interval   = config.evaluation.number_of_episodes_before_eval,
            outdir=outdir,
            max_episode_len=max_episode_len,
            step_offset=step_offset,
            evaluation_hooks=evaluation_hooks,
            save_best_so_far_agent=save_best_so_far_agent,
            logger=logger,
        )
        if use_tensorboard:
            evaluator.start_tensorboard_writer(outdir, stop_event)

    if random_seeds is None:
        random_seeds = np.arange(config.number_of_processes)

    @print.indent.function
    def run_func(process_idx):
        config.verbose and print(f"[starting process{process_idx} (run_func())]")
        random_seed.set_random_seed(random_seeds[process_idx])

        env = make_env(process_idx, test=False)
        if evaluator is None:
            eval_env = env
        else:
            eval_env = make_env(process_idx, test=True)
        if make_agent is not None:
            local_agent = make_agent(process_idx)
            if use_shared_memory:
                for attr in agent.shared_attributes:
                    setattr(local_agent, attr, getattr(agent, attr))
        else:
            local_agent = agent
        
        local_agent.process_idx = process_idx
        local_agent.is_malicious = process_is_malicious[process_idx]

        f = lambda : inner_training_loop(
                process_idx=process_idx,
                number_of_episodes=number_of_episodes,
                agent=local_agent,
                env=env,
                max_number_of_episodes=steps,
                outdir=outdir,
                max_episode_len=max_episode_len,
                evaluator=evaluator,
                stop_event=stop_event,
                exception_event=exception_event,
                eval_env=eval_env,
                global_step_hooks=global_step_hooks,
                logger=logger,
                individual_updates_ready_barrier=individual_updates_ready_barrier,
                individual_updates_contributed_barrier=individual_updates_contributed_barrier,
                process_permabanned=process_permabanned,
                process_median_episode_rewards=process_median_episode_rewards,
            )
        try:
            if profile:
                import cProfile
                cProfile.runctx("f()", globals(), locals(), f"profile-{os.getpid()}.out")
            else:
                f()
        except Exception as error:
            raise error
        finally:
            env.close()
            if eval_env and eval_env is not env:
                eval_env.close()
    
    config.verbose and print("[about to call async_.run_async()]")
    async_.run_async(config.number_of_processes, run_func)
    config.verbose and print("[done calling async_.run_async()]")
    
    # make sure the process_median_episode_rewards is valid before stopping
    for process_index in range(config.number_of_processes):
        if process_permabanned[process_index]:
            process_median_episode_rewards[process_index] = 0
    stop_event.set()

    if evaluator is not None and use_tensorboard:
        evaluator.join_tensorboard_writer()

def mp_to_numpy(mp_arr):
    return np.asarray(list(mp_arr))