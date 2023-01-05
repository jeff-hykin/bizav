import logging
import os
import sys
from random import random, sample, choices
from statistics import mean
import math
import json

import gym
import optuna
import numpy as np
import torch
from torch import nn
from super_map import LazyDict
from blissful_basics import singleton, max_indices, print, to_pure, shuffled, normalize, countdown, randomly_pick_from

from pfrl.experiments.evaluator import AsyncEvaluator
from pfrl.utils import async_, random_seed, copy_param

from main import utils
from main.config import config, env_config, info
from main.utils import trend_calculate

# 
# constants
# 
debug = config.debug
check_rate = config.number_of_processes
should_log = countdown(config.log_rate)
show_iterations = True

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
    main_agent=None,
    agents=None,
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
    if show_iterations:
        from informative_iterator import ProgressBar
        progress_iter = iter(ProgressBar(config.training.episode_count, title="trial run"))
    config.verbose and print("[starting middle_training_function()]\n")
    for hook in evaluation_hooks:
        if not hook.support_middle_training_function:
            raise ValueError("{} does not support middle_training_function().".format(hook))
    
    evaluator = None if eval_interval is None else AsyncEvaluator(
        n_steps=eval_n_steps,
        n_episodes=eval_n_episodes,  # eval_n_episodes = config.evaluation.number_of_epsiodes_during_eval,
        eval_interval=eval_interval, # eval_interval   = config.evaluation.number_of_episodes_before_eval,
        outdir=outdir,
        max_episode_len=max_episode_len,
        step_offset=step_offset,
        evaluation_hooks=evaluation_hooks,
        save_best_so_far_agent=save_best_so_far_agent,
        logger=logger or logging.getLogger(__name__),
    )
    
    # 
    # reset globals
    # 
    if True:
        use_max_defence      = config.defense_method == 'max'
        use_softmax_defense  = config.defense_method == 'softmax'
        use_permaban_defense = config.defense_method == 'ucb'
        use_no_defense       = not (use_softmax_defense or use_max_defence or use_permaban_defense)
        
        # 
        # init shared values
        # 
        shared = LazyDict(
            prev_total_number_of_episodes   = -1,
            number_of_timesteps             = 0,
            number_of_episodes              = 0,
            number_of_updates               = 0, # Number of total rollouts completed
            filtered_count                  = 0, # Number of permanently filtered agents
            successfully_filtered_sum       = 0,
            successfully_filtered_increment = 0,
            latest_eval_score               = 0,
            episode_reward_trend            = [],
        )

    # 
    # wrapper for processes
    # 
    envs, group = unvectorize(
        gym.vector.make(
            config.env_config.env_name,
            asynchronous=False,
            num_envs=config.number_of_processes
        )
    )
    malicious_indices = shuffled(list(range(config.number_of_processes)))[0:config.number_of_malicious_processes]
    class Process:
        indices_of_malicious            = malicious_indices
        q_values                        = [0]*config.number_of_processes
        temp_banned_count               = [1]*config.number_of_processes # ASK: setting to 1 avoids a division by 0, but treats all processes equal
        temp_ban                        = [0]*config.number_of_processes
        accumulated_normalized_distance = [0]*config.number_of_processes
        malicious                       = [ (1 if index in malicious_indices else 0) for index in range(config.number_of_processes) ]
        
        def __init__(self, process_index, agent, evaluator, env):
            config.verbose and print(f"[creating Process({process_index})]")
            
            self.index = process_index
            
            self.env                = env
            self.evaluator          = evaluator
            self.agent              = agent
            self.agent.process_idx  = process_index
            self.agent.is_malicious = Process.malicious[process_index]
            
            self.is_permabanned        = 0
            self.accumulated_suspicion = 0
            self.latest_episode_reward = 0
            self.median_episode_reward = 0
        
            self.on_new_episode()
        
        def on_new_episode(self):
            self.episode_reward  = 0
            self.episode_len     = 0
            self.number_of_timesteps_for_this_episode = 0
            shared.number_of_episodes += 1
        
        def choose_action(self, observation, reward, done, info):
            action = self.agent.act(observation)
            # first observation
            if type(reward) == type(None):
                self.episode_reward  = 0
                self.episode_len     = 0
                self.number_of_timesteps_for_this_episode = 0
            else:
                self.number_of_timesteps_for_this_episode += 1
                self.episode_reward += reward
                self.episode_len += 1
                reset = self.episode_len == max_episode_len or info.get("needs_reset", False)
                self.agent.observe(observation, reward, done, reset)
                if done:
                    self.on_new_episode()
            
            self.env.choose_action(action)
            return action
        
        @property
        def gradient(self):
            my_grad = []
            for param in self.agent.model.parameters():
                if param.grad is not None:
                    grad_np = param.grad.detach().clone().numpy().flatten()
                else:
                    grad_np = np.zeros(param.size(), dtype=np.float).flatten()
                for j in range(len(grad_np)):
                    my_grad.append(grad_np[j])
            return np.asarray(my_grad)
        
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
        def accumulated_distance(self): return Process.accumulated_normalized_distance[self.index]
        @accumulated_distance.setter
        def accumulated_distance(self, value): Process.accumulated_normalized_distance[self.index] = value
        
        # 
        # temp ban
        # 
        @property
        def is_temp_banned(self):
            if Process.temp_ban[self.index]:
                return True
            return False
        @is_temp_banned.setter
        def is_temp_banned(self, value): Process.temp_ban[self.index] = value+0
        
        # 
        # temp ban
        # 
        @property
        def is_malicious(self):
            if Process.malicious[self.index]:
                return True
            
            return False
        
        # 
        # q value
        # 
        @property
        def q_value(self): return Process.q_values[self.index]
        @q_value.setter
        def q_value(self, value): Process.q_values[self.index] = value
        
        # 
        # visits
        # 
        @property
        def temp_ban_count(self): return Process.temp_banned_count[self.index]
        @temp_ban_count.setter
        def temp_ban_count(self, value): Process.temp_banned_count[self.index] = value
        
        def distance_metric(self, all_grads, kind):
            this_gradient = torch.tensor(all_grads[self.index])
            all_other_gradients = []
            indices_of_others = []
            for process, each_gradient in zip(processes, all_grads):
                if process.index != self.index:
                    all_other_gradients.append(each_gradient)
                    indices_of_others.append(process.index)
            all_other_gradients = torch.tensor(np.vstack(all_other_gradients))
            
            if kind == 'mean_all':
                process_distances = euclidean_dist(
                    torch.vstack([ this_gradient ]),
                    all_other_gradients,
                )
                return float(process_distances[0].mean())
            
            elif kind == 'sum_all':
                process_distances = euclidean_dist(
                    torch.vstack([ this_gradient ]),
                    all_other_gradients,
                )
                return float(process_distances[0].sum())
                
            elif kind == 'median_point':
                median = all_other_gradients.median(axis=0).values
                process_distances = euclidean_dist(
                    torch.vstack([ this_gradient ]),
                    torch.vstack([ torch.tensor(median) ])
                )
                return float(process_distances[0].sum())
                
            elif kind == 'mean_point':
                mean = all_other_gradients.mean(axis=0)
                process_distances = euclidean_dist(
                    torch.vstack([ this_gradient ]),
                    torch.vstack([ torch.tensor(mean) ]),
                )
                    
                return float(process_distances[0].sum())
    # 
    # create all processes
    # 
    processes = tuple(
        Process(each_index, agent=agent, evaluator=evaluator, env=env)
            for each_index, (agent, env) in enumerate(zip(agents, envs))
    )
    
    # 
    # create UCB manager
    # 
    @singleton
    class ucb:
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
            
            if use_max_defence:
                distances = []
                for each_process_being_reviewed in processes:
                    distance = each_process_being_reviewed.distance_metric(all_grads, kind=config.distance_kind)
                    debug and print(f'''math.log(distance) = {math.log(distance)}, is_malicious: {each_process_being_reviewed.is_malicious}''')
                    distances.append(distance)
                
                # 
                # accumulate distance
                # 
                weights = force_sum_to_one(distances, minimum=0)
                for each_process_being_reviewed, normalized_distance in zip(processes, weights):
                    each_process_being_reviewed.accumulated_distance += normalized_distance * config.number_of_processes
            
            if use_softmax_defense:
                reward_for_each_temp_banned_index = []
                distances = []
                for each_process_being_reviewed in processes:
                    distance = each_process_being_reviewed.distance_metric(all_grads, kind=config.distance_kind)
                    # debug and print(f'''distance = {distance}''')
                    debug and print(f'''math.log(distance) = {math.log(distance)}, is_malicious: {each_process_being_reviewed.is_malicious}''')
                    distances.append(distance)
                    flipped_value = -distance
                    ucb_reward = config.env_config.variance_scaling_factor * flipped_value
                    reward_for_each_temp_banned_index.append(ucb_reward)
                
                
                if debug:
                    debugging_distances = []
                    for index, each_distance in enumerate(distances):
                        process = processes[index]
                        debugging_distances.append((index, each_distance, process.is_malicious, False))
                        grads_copy = torch.tensor(all_grads.tolist())
                        grads_copy[index] *= -2.5
                        debugging_distances.append((index, process.distance_metric(grads_copy, kind=config.distance_kind), process.is_malicious, True))
                    
                    for index, distance, is_malicious, is_malicious_flipped in sorted(debugging_distances, key=lambda each: each[1]):
                        if is_malicious_flipped and is_malicious:
                            print(f'''process{index} distance is {round(float(distance), 1)}, is_malicious_flipped={is_malicious_flipped}''')
                        elif is_malicious:
                            print(f'''process{index} distance is {round(float(distance), 1)}, is_malicious={is_malicious}''')
                        elif is_malicious_flipped:
                            print(f'''process{index} distance is {round(float(distance), 1)}, is_flipped={is_malicious_flipped}''')
                        else:
                            print(f'''process{index} distance is {round(float(distance), 1)}''')
                
                # 
                # accumulate distance
                # 
                weights = force_sum_to_one(distances, minimum=0)
                for each_process_being_reviewed, normalized_distance in zip(processes, weights):
                    each_process_being_reviewed.accumulated_distance += normalized_distance * config.number_of_processes
                
                # 
                # accumulate suspicion
                # 
                current_suspicion_for = [ each * config.number_of_malicious_processes for each in force_sum_to_one(list(to_pure(Process.accumulated_normalized_distance))) ]
                for process in processes:
                    process.accumulated_suspicion *= 1 + current_suspicion_for[process.index]
                
                return reward_for_each_temp_banned_index
        
        def get_uncertainty(self, values):
            values = force_sum_to_one(values)
            output = sum(sorted(values)[:-config.expected_number_of_malicious_processes])/len(values)
            if output != output: # Nan error
                return 0
            else:
                return output
            
        def update_step(self, ucb_reward):
            
            if use_softmax_defense:
                pass
            elif use_max_defence:
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
            np_process_is_malicious      = np.asarray(Process.malicious)
            np_process_temp_ban          = np.asarray(Process.temp_ban)
            np_process_temp_banned_count = np.asarray(Process.temp_banned_count)
            np_value_per_process         = np.asarray(Process.q_values)
            
            if use_permaban_defense:
                # Get the true UCB t value
                ucb_timesteps = np.sum(np_process_temp_banned_count) - (env_config.permaban_threshold+1) * shared.filtered_count
                successful_bans = 0
                ban_summary = []
                for process in processes:
                    if process.is_malicious:
                        if process.is_temp_banned or process.is_permabanned:
                            successful_bans += 1
                            ban_summary.append('y-m') # yes-malicious (good)
                        else:
                            ban_summary.append('n-m') # no-malicious
                    else:
                        if process.is_temp_banned or process.is_permabanned:
                            ban_summary.append('y-b') # yes-bengin
                        else:
                            ban_summary.append('n-b') # no-bengin (good)
                        
                # Compute UCB policy values (Q-value + uncertainty)
                config.verbose and print(json.dumps(dict(
                    step=shared.number_of_updates,
                    successfully_filtered=successful_bans,
                    ban_summary=ban_summary,
                    filter_choice=Process.temp_ban,
                    temp_banned_count=list(np.round(np_process_temp_banned_count, 2)),
                    q_vals=list(np.round(np_value_per_process, 3)),
                )))
                return np_value_per_process + np.sqrt((np.log(ucb_timesteps)) / np_process_temp_banned_count)
            
            if use_softmax_defense or use_max_defence:
                if sum(Process.temp_ban) == 0:
                    previously_chosen_indices = tuple()
                else:
                    previously_chosen_indices = max_indices(Process.temp_ban)
                
                number_of_banned_items = (config.expected_number_of_malicious_processes * shared.number_of_updates)
                ucb_timesteps = np.sum(np_process_temp_banned_count) - number_of_banned_items
                # Compute UCB policy values (Q-value + uncertainty)
                return np_value_per_process + np.sqrt((np.log(ucb_timesteps)) / np_process_temp_banned_count)
                
        def choose_action(self):
            output = None
            if use_permaban_defense:
                number_permabanned = sum(1 for each in processes if each.is_permabanned)
                # already banned everyone
                if number_permabanned >= expected_number_of_malicious_processes:
                    output = []
                else:
                    # Initial selection (Process.temp_banned_count 0) #TODO properly select at random
                    np_process_temp_banned_count = np.asarray(Process.temp_banned_count)
                    if np.min(np_process_temp_banned_count) < 1:
                        output = [ np.argmin(np_process_temp_banned_count) ]
                    else:
                        # multiple can have highest score, so choose randomly between them
                        random_first_place = randomly_pick_from(max_indices(ucb.value_of_each_process()))
                        output = [ random_first_place ]
            
            elif use_max_defence:
                weights = list(Process.accumulated_normalized_distance)
                sorted_indicies_and_distances = sorted(list(enumerate(weights)), reverse=True, key=lambda each: each[1])
                debug and print(f'''sorted_indicies_and_distances = {sorted_indicies_and_distances}''')
                indicies_with_biggest_distance = [ index for index, distance in sorted_indicies_and_distances ]
                output = indicies_with_biggest_distance[:config.expected_number_of_malicious_processes]
                
                # 
                # logging
                #
                if debug: 
                    for index, _ in sorted_indicies_and_distances:
                        process = processes[index]
                        print(f'Process({index}):')
                        print(f'''    accumulated_distance:{process.accumulated_distance}''')
                        print(f'''    is_malicious:{process.is_malicious}''')
                        print(f'''    is_central_agent:{process.is_central_agent}''')
                np_process_is_malicious      = np.asarray(Process.malicious)
                np_process_temp_ban          = np.asarray(Process.temp_ban)
                np_process_temp_banned_count = np.asarray(Process.temp_banned_count)
                successful = sum(np_process_is_malicious * Process.temp_ban)
                
                shared.successfully_filtered_sum += successful
                shared.successfully_filtered_increment += 1
                
                malicious = list(Process.malicious) 
                log_weights = [ round(math.log(each+1), 3) for each in weights ]
                
                normalized_log_weights = force_sum_to_one(log_weights)
                malicious_log_weight     = sorted([ round(log_weight*config.number_of_processes, 1)/2 for each_process, log_weight in zip(processes, normalized_log_weights) if     each_process.is_malicious ])
                non_malicious_log_weight = sorted([ round(log_weight*config.number_of_processes, 1)/2 for each_process, log_weight in zip(processes, normalized_log_weights) if not each_process.is_malicious ])
                
                config.verbose and print(json.dumps(dict(
                    step=shared.number_of_updates,
                    number_of_episodes=shared.number_of_episodes,
                    number_of_timesteps=shared.number_of_timesteps,
                    successfully_filtered_avg=round(shared.successfully_filtered_sum/shared.successfully_filtered_increment, 2),
                    successfully_filtered=successful,
                    processes={
                        f"{each_index}": dict(is_malicious=is_malicious+0, filtered=filtered, log_weight=round(each_log_weight, 2), weight=round(each_weight))
                            for each_index, (is_malicious, each_weight, each_log_weight, filtered) in enumerate(zip(malicious, weights, log_weights, Process.temp_ban))
                    },
                    temp_banned_count=list(np.round(np_process_temp_banned_count, 2)),
                )))
            elif use_softmax_defense:
                debug and print(f'''Process.accumulated_normalized_distance = {list(Process.accumulated_normalized_distance)}''')
                suspicions = [ each.accumulated_suspicion for each in processes ]
                debug and print(f'''raw suspicions = {sorted(suspicions,reverse=True)}''')
                process_indices = list(range(config.number_of_processes))
                
                uncertainty = ucb.get_uncertainty(suspicions)
                config.verbose and print(f'''{{ "uncertainty": {uncertainty} }}''')
                weights = force_sum_to_one(suspicions)
                debug and print(f'''normalized weights = {sorted(weights, reverse=True)}''')
                weights = [ each + uncertainty for each in weights ] # this operation is done so that the min isn't always a 0-weight
                debug and print(f'''normalized weights with uncertainty = {sorted(weights, reverse=True)}''')
                
                picked_processes = random_choose_k(items=processes, weights=weights, k=config.expected_number_of_malicious_processes)
                
                # who to ban this round
                output = [ each.index for each in picked_processes ]
                debug and print(f'''output = {output}''')
            
                np_process_is_malicious      = np.asarray(Process.malicious)
                np_process_temp_ban          = np.asarray(Process.temp_ban)
                np_process_temp_banned_count = np.asarray(Process.temp_banned_count)
                np_value_per_process         = np.asarray(Process.q_values)
                successful = sum(np_process_is_malicious * Process.temp_ban)
                
                shared.successfully_filtered_sum += successful
                shared.successfully_filtered_increment += 1
                
                malicious = list(Process.malicious) 
                log_weights = [ round(math.log(each+1), 3) for each in list(Process.accumulated_normalized_distance) ]
                
                normalized_log_weights = force_sum_to_one(log_weights)
                malicious_log_weight     = sorted([ round(log_weight*config.number_of_processes, 1)/2 for each_process, log_weight in zip(processes, normalized_log_weights) if     each_process.is_malicious ])
                non_malicious_log_weight = sorted([ round(log_weight*config.number_of_processes, 1)/2 for each_process, log_weight in zip(processes, normalized_log_weights) if not each_process.is_malicious ])
                    
                config.verbose and print(json.dumps(dict(
                    step=shared.number_of_updates,
                    number_of_episodes=shared.number_of_episodes,
                    number_of_timesteps=shared.number_of_timesteps,
                    successfully_filtered_avg=round(shared.successfully_filtered_sum/shared.successfully_filtered_increment, 2),
                    successfully_filtered=successful,
                    malicious_log_weight=malicious_log_weight,
                    non_malicious_log_weight=non_malicious_log_weight,
                    processes={
                        f"{each_index}": dict(is_malicious=is_malicious+0, filtered=filtered, log_weight=round(each_log_weight, 2), weight=round(each_weight))
                            for each_index, (is_malicious, each_weight, each_log_weight, filtered) in enumerate(zip(malicious, Process.accumulated_normalized_distance, log_weights, Process.temp_ban))
                    },
                    temp_banned_count=list(np.round(np_process_temp_banned_count, 2)),
                    q_vals=list(np.round(np_value_per_process, 3)),
                )))
            
            return output
                
        @property
        def gradient_of_agents(self):
            all_grads = []
            for process in processes:
                all_grads.append(process.gradient)
            
            if debug:
                print("all_grads")
                with print.indent:
                    for each in all_grads:
                        print(f'''each.sum() = {each.sum()}''')
            
            return np.vstack(all_grads)
    
    # 
    # 
    # Run all processes
    # 
    # 
    while True:
        # get actions, which auto-triggers the next step of the env
        for each in processes:
            each.choose_action(*each.env.step_data)
        
        shared.number_of_timesteps += len(processes)
        if all(each.agent.updated for each in processes):
            shared.number_of_updates += 1
            for each in processes:
                each.agent.updated = False
            
            # 
            # update step
            # 
            with print.indent: # .block(f"update step {shared.number_of_episodes}/{config.training.episode_count}")
                all_grads = np.vstack([ each.gradient for each in processes ])
                debug and print("started individual_updates_ready_barrier()")
                debug and print("starting early_stopping_check()")
                
                all_malicious_actors_found = shared.filtered_count == config.expected_number_of_malicious_processes
                if all_malicious_actors_found:
                    debug and print("finished individual_updates_ready_barrier()")
                    return
                
                # Update values
                debug and print("shared.number_of_updates", shared.number_of_updates)
                if shared.number_of_updates != 0:
                    # Compute gradient mean
                    debug and print("starting reward_func()")
                    ucb_reward = ucb.reward_func(ucb.gradient_of_agents)
                    
                    # Update Q-values
                    debug and print("starting update_step()")
                    ucb.update_step(ucb_reward)
                    
                    if use_permaban_defense:
                        # 
                        # Permanently disable an self.agent
                        # 
                        prev_amount_filtered = shared.filtered_count
                        for process_index, ban_count in enumerate(Process.temp_banned_count):
                            if ban_count >= env_config.permaban_threshold:
                                processes[process_index].is_permabanned = 1
                        shared.filtered_count = sum(1 for process in processes if process.is_permabanned)
                        
                        # 
                        # if someone was just recently permabaned
                        # 
                        if prev_amount_filtered < shared.filtered_count:
                            # Reset non-disabled Process.temp_banned_count/Q-values
                            for process_index, ban_count in enumerate(Process.temp_banned_count):
                                if ban_count < env_config.permaban_threshold:
                                    Process.temp_banned_count[process_index] = 0
                                    Process.q_values[process_index] = 0
                
                # Select next ucb action
                debug and print("starting choose_action()")
                who_to_ban = ucb.choose_action()
                debug and print(f"finished choose_action(), who_to_ban={who_to_ban}")
                if who_to_ban:
                    for each_process in processes:
                        if each_process.index in who_to_ban:
                            each_process.temp_ban_count += 1
                            Process.temp_ban[each_process.index] = 1 # aka True
                        else:
                            Process.temp_ban[each_process.index] = 0
                
                debug and print("finished individual_updates_ready_barrier()")
        
            # 
            # contribute update
            # 
            for each in processes:
                if not each.is_banned:
                    debug and print(f'''self.agent{each.index}.add_update()''')
                    debug and print(f'''adding update, each.is_malicious = {each.is_malicious}''')
                    # include each.agent's gradient in global model
                    copy_param.add_grad(target_link=main_agent.model, source_link=each.agent.model)
        
        # 
        # sync_updates()
        # 
        if True:
            if shared.filtered_count != config.expected_number_of_malicious_processes:
                num_updates = config.number_of_processes - (shared.filtered_count + 1)
            else:
                num_updates = config.number_of_processes - shared.filtered_count
            
            for param in main_agent.model.parameters():
                if param.grad is not None:
                    param.grad = param.grad / num_updates
            
            main_agent.optimizer.step()
            
            # deliver the gradient change to all the agents
            for process in processes:
                copy_param.copy_param(target_link=process.agent.model, source_link=main_agent.model)
                process.agent.after_update()
        
        # 
        # stop condition
        # 
        if shared.number_of_episodes >= config.training.episode_count:
            break
    
    config.verbose and print("[all processes finished]")

def force_sum_to_one(values, minimum=None):
    values = tuple(values) # for iterators
    count = len(values)
    if count == 0:
        return []
    
    from statistics import mean as average
    minimum = minimum if minimum != None else min(values)
    positive_values = tuple(each-minimum for each in values)
    sum_total = sum(positive_values)
    if sum_total == 0:
        return [1/count]*count
    else:
        return [ each/sum_total for each in positive_values ]

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

def random_choose_k(items, weights, k):
    # NOTE: a statistician probably wouldnt be very happy with this function. And it could probably be improved
    # chooses k UNIQUE elements from a set of items, each of which have their own weight
    choices = set()
    remaining_indices_and_weights = list(enumerate(weights))
    import random
    while len(choices) < k:
        remaining_indices_and_weights = [ (index, weight) for index, weight in remaining_indices_and_weights if index not in choices ]
        remaining_indicies = [ index for index, weight in remaining_indices_and_weights ]
        remaining_weights  = [ weight for index, weight in remaining_indices_and_weights ]
        if sum(remaining_weights) == 0:
            for each in shuffled(remaining_indicies)[:(k - len(choices))]:
                choices.add(each)
            break
        random_index = random.choices(
            remaining_indicies,
            weights=remaining_weights,
            k=1,
        )[0]
        choices.add(random_index)
    
    return [ items[index] for index in choices ]


def unvectorize(env):
    vectorized_env = env
    action_ready = 0
    all_components = []
    @singleton
    class group:
        observations = tuple(iter(vectorized_env.reset()))
        rewards = [None]*len(observations)
        dones = [None]*len(observations)
        infos = [None]*len(observations)
        
        @property
        def actions(self):
            return tuple(each._qued_action for each in all_components)
            
        @property
        def collective_step_ready(self):
            return all(type(action) != type(None) for action in self.actions)
        
        def reset(self):
            self.observations = tuple(iter(vectorized_env.reset()))
            return self.observations
        
        def step(self):
            group.observations, group.rewards, group.dones, group.infos = vectorized_env.step(actions=self.actions)
            group.observations = tuple(iter(group.observations))
            group.rewards      = tuple(iter(group.rewards))
            group.dones        = tuple(iter(group.dones))
            group.infos        = tuple(iter(group.infos))
            # clear actions
            for each in all_components:
                each._qued_action = None
            
            return group.observations, group.rewards, group.dones, group.infos
    
    class UnvectorizedEnv(gym.Env):
        action_space      = env.single_action_space
        observation_space = env.single_observation_space
        
        def __init__(self, index):
            self._index = index
            self._source = vectorized_env
            self._qued_action = None
            all_components.append(self)
        
        @property
        def step_data(self):
            return group.observations[self._index], group.rewards[self._index], group.dones[self._index], group.infos[self._index]
        
        def choose_action(self, action):
            self._qued_action = action
            actions = tuple(each._qued_action for each in all_components)
            if all(type(action) != type(None) for action in actions):
                group.observations, group.rewards, group.dones, group.infos = vectorized_env.step(actions=actions)
                group.observations = tuple(iter(group.observations))
                group.rewards      = tuple(iter(group.rewards))
                group.dones        = tuple(iter(group.dones))
                group.infos        = tuple(iter(group.infos))
                # clear actions
                for each in all_components:
                    each._qued_action = None
        
        def step(self, action):
            self.choose_action(action)
            return self.step_data
            
        def reset(self):
            return group.observations[self._index]
    
    envs = tuple(UnvectorizedEnv(index) for index in range(len(group.observations)))
    return envs, group