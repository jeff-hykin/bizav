import logging
import os
import signal
import subprocess
import sys
from random import random, sample, choices
from statistics import mean
import math

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn
from super_map import LazyDict

from pfrl.experiments.evaluator import AsyncEvaluator
from pfrl.utils import async_, random_seed

from main import utils
from main.config import config, env_config, info
from main.utils import trend_calculate

# 
# constants
# 
check_rate = config.number_of_processes

# 
# globals
# 
central_agent_process_index    = None
when_all_processes_are_updated = None
prev_total_number_of_episodes  = None
number_of_timesteps            = None
number_of_episodes             = None
number_of_updates              = None
process_index_to_temp_filter   = None
filtered_count                 = None
act_val                        = None
visits                         = None
filtered_agents                = None
episode_reward_trend           = None
median_episode_rewards         = None

def reset_globals():
    global central_agent_process_index, when_all_processes_are_updated, prev_total_number_of_episodes, number_of_timesteps, number_of_episodes, number_of_updates,  process_index_to_temp_filter,  filtered_count,  act_val,  visits,  filtered_agents, episode_reward_trend, median_episode_rewards
    episode_reward_trend = []
    central_agent_process_index = sample(list(range(config.number_of_processes)), k=1)[0]
    when_all_processes_are_updated = None
    prev_total_number_of_episodes = -1
    
    # 
    # init shared values
    # 
    number_of_timesteps          = mp.Value("l", 0)
    number_of_episodes           = mp.Value("l", 0)
    number_of_updates            = mp.Value("l", 0) # Number of total rollouts completed
    process_index_to_temp_filter = mp.Value("i", 0) # UCB action 'broadcast'
    filtered_count               = mp.Value("l", 0) # Number of permanently filtered agents

    act_val                = mp.Array("d", config.number_of_processes) # Q-values
    visits                 = mp.Array("d", config.number_of_processes) # number of visits
    filtered_agents        = mp.Array("l", config.number_of_processes) # Permanently filtered agent memory
    median_episode_rewards = mp.Array("d", config.number_of_processes)
    for process_index in range(config.number_of_processes):
        act_val[process_index] = 0
        visits[process_index]            = 0
        filtered_agents[process_index]   = 0


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

def train_loop(
    process_idx,
    env,
    agent,
    max_number_of_episodes,
    outdir,
    number_of_episodes,
    stop_event,
    exception_event,
    median_episode_rewards,
    max_episode_len=None,
    evaluator=None,
    eval_env=None,
    successful_score=None,
    logger=None,
    global_step_hooks=[],
    all_updated_barrier=None,
    update_barrier=None,
    process_index_to_temp_filter=None,
    filtered_agents=None,
    byzantine_agent_number=0,
):
    global episode_reward_trend
    config.verbose and print("[starting train_loop()]")
    logger = logger or logging.getLogger(__name__)

    if eval_env is None:
        eval_env = env

    def save_model():
        if process_idx == 0:
            # Save the current model before being killed
            dirname = os.path.join(outdir, "{}_except".format(global_episode_count))
            agent.save(dirname)
            logger.info("Saved the current model to %s", dirname)

    try:
        config.verbose and print(f'''env = {env}''')
        episode_r                   = 0
        global_episode_count        = 0
        local_t                     = 0
        global_episodes             = 0
        obs                         = env.reset()
        episode_len                 = 0
        successful                  = False
        number_of_timesteps_for_this_episode = 0

        while True:
            # a_t
            a = agent.act(obs)
            # o_{t+1}, r_{t+1}
            obs, r, done, info = env.step(a)
            number_of_timesteps_for_this_episode += 1
            local_t += 1
            episode_r += r
            episode_len += 1
            reset = episode_len == max_episode_len or info.get("needs_reset", False)
            agent.observe(obs, r, done, reset)

            if agent.updated:
                all_updated_barrier.wait()  # Wait for all agents to complete rollout, then run when_all_processes_are_updated()
                # only one process runs this
                if process_idx == central_agent_process_index:
                    when_all_processes_are_updated()
                    
                if byzantine_agent_number > 0:
                    # If not current UCB action and not permanently filtered, include agent's gradient in global model
                    if process_index_to_temp_filter.value != process_idx and filtered_agents[process_idx] == 0: agent.add_update()
                else:
                    agent.add_update()
                update_barrier.wait()   # Wait for all agents to contribute their gradients to global model, the sync_updates() to step it's optimizer
                agent.after_update()    # Each agent will download the global model after the optimizer max_number_of_episodes

            if done or reset:# Get and increment the global number_of_episodes
                with number_of_episodes.get_lock():
                    median_episode_rewards[process_idx] = agent.median_reward_per_episode
                    number_of_episodes.value += 1
                    number_of_timesteps.value += number_of_timesteps_for_this_episode
                    global_episodes = number_of_episodes.value
                    global_episode_count = number_of_episodes.value
                
                # reset
                number_of_timesteps_for_this_episode = 0
                
                for hook in global_step_hooks:
                    hook(env, agent, global_episode_count)

                metric_line = str(global_episode_count)+'; '+str(process_idx)+'; '+str(episode_r)+'; '
                stats = agent.get_statistics()
                for item in stats:
                    metric_line += str(item) + '; '
                if stats[0] != -1:
                    logger.info(metric_line[:-2])

                # Evaluate the current agent
                if evaluator is not None:
                    eval_score = evaluator.evaluate_if_necessary(
                        t=global_episode_count, episodes=global_episodes, env=eval_env, agent=agent
                    )

                    if (
                        eval_score is not None
                        and successful_score is not None
                        and eval_score >= successful_score
                    ):
                        stop_event.set()
                        successful = True
                        # Break immediately in order to avoid an additional
                        # call of agent.act_and_train
                        break
                
                proportional_number_of_timesteps = number_of_timesteps.value / config.number_of_processes
                if global_episode_count >= max_number_of_episodes or stop_event.is_set():
                    break

                # Start a new episode
                episode_r = 0
                episode_len = 0
                obs = env.reset()

            if process_idx == 0 and exception_event.is_set():
                logger.exception("An exception detected, exiting")
                save_model()
                kill_all()
        
        update_barrier.abort()
        all_updated_barrier.abort()

    except (Exception, KeyboardInterrupt):
        save_model()
        raise

    if global_episode_count == max_number_of_episodes:
        # Save the final model
        dirname = os.path.join(outdir, "{}_finish".format(max_number_of_episodes))
        agent.save(dirname)
        logger.info("Saved the final agent to %s", dirname)

    if successful:
        # Save the successful model
        dirname = os.path.join(outdir, "successful")
        agent.save(dirname)
        logger.info("Saved the successful agent to %s", dirname)

def train_agent_async(
    outdir,
    processes,
    make_env,
    profile=False,
    steps=8 * 10**7,
    eval_interval=10**6,
    eval_n_steps=None,
    eval_n_episodes=10,
    eval_success_threshold=0.0,
    max_episode_len=None,
    step_offset=0,
    successful_score=None,
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
    num_agents_byz=0,
    permaban_threshold=1,
    output=LazyDict(),
):
    """Train agent asynchronously using multiprocessing.

    Either `agent` or `make_agent` must be specified.

    Args:
        outdir (str): Path to the directory to output things.
        processes (int): Number of processes.
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
        successful_score (float): Finish training if the mean score is greater
            or equal to this value if not None
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
    global central_agent_process_index, when_all_processes_are_updated, prev_total_number_of_episodes, number_of_timesteps, number_of_episodes, number_of_updates,  process_index_to_temp_filter,  filtered_count,  act_val,  visits,  filtered_agents, episode_reward_trend, median_episode_rewards, episode_reward_trend
    config.verbose and print("[starting train_agent_async()]")
    logger = logger or logging.getLogger(__name__)

    for hook in evaluation_hooks:
        if not hook.support_train_agent_async:
            raise ValueError("{} does not support train_agent_async().".format(hook))
    
    # Prevent numpy from using multiple threads
    os.environ["OMP_NUM_THREADS"] = "1"
    
    reset_globals()
    
    output.update(dict(
        median_episode_rewards=median_episode_rewards,
        episode_reward_trend=episode_reward_trend,
        check_rate=check_rate,
        number_of_episodes=0,
    ))
    
    # 
    # create UCB manager
    # 
    @singleton
    class ucb:
        value_per_process = act_val
        
        def reward_func(self, agent_gradients):
            agent_gradients = torch.tensor(agent_gradients)
            # agent_variance = np.var(agent_gradients, axis=-1)
            average_variance = euclidean_dist(agent_gradients, agent_gradients).mean()
            flipped_value = -average_variance
            ucb_reward = config.env_config.variance_scaling_factor * flipped_value
            return ucb_reward
        
        def update_step(self, ucb_reward):
            process_index    = process_index_to_temp_filter.value
            old_q_value      = ucb.value_per_process[process_index]
            number_of_visits = visits[process_index]
            change_in_value  = (ucb_reward - old_q_value) / number_of_visits
            # apply the new value
            ucb.value_per_process[process_index] += change_in_value
        
        def choose_action(self):
            # Mask permanently filtered agents
            for process_index, process_is_filtered in enumerate(filtered_agents):
                if process_is_filtered:
                    import math
                    ucb.value_per_process[process_index] = -math.inf
            
            np_visits = mp_to_numpy(visits)
            np_value_per_processs = mp_to_numpy(ucb.value_per_process)
            
            config.verbose and print('Step', number_of_updates.value, process_index_to_temp_filter.value, "visits", list(np.round(np_visits, 2)), " episode_count:", number_of_episodes.value, "q_vals:", list(np.round(np_value_per_processs, 3)), end="\r")
            
            # Get the true UCB t value
            ucb_timesteps = np.sum(np_visits) - (env_config.permaban_threshold+1) * filtered_count.value
            # Compute UCB policy values (Q-value + uncertainty)
            values = np_value_per_processs + np.sqrt((np.log(ucb_timesteps)) / np_visits)
            
            # Initial selection (visits 0) #TODO properly select at random
            if np.min(np_visits) < 1:
                return np.argmin(np_visits)
            else:
                return np.argmax(values)
        
        @property
        def smart_gradient_of_agents(self):
            all_grads = []
            for process_index in range(config.number_of_processes):
                my_grad = []
                agent_is_temp_filtered      = process_index_to_temp_filter.value == process_index
                agent_is_permantly_filtered = filtered_agents[process_index] == 1
                # If filtered, don't include in gradient mean
                if agent_is_temp_filtered or agent_is_permantly_filtered:
                    continue
                for param in agent.local_models[process_index].parameters():
                    if param.grad is not None:
                        grad_np = param.grad.detach().clone().numpy().flatten()
                    else:
                        grad_np = np.zeros(param.size(), dtype=np.float).flatten()
                    for j in range(len(grad_np)):
                        my_grad.append(grad_np[j])
                all_grads.append(np.asarray(my_grad))
            return np.vstack(all_grads)
        
    global when_all_processes_are_updated
    def when_all_processes_are_updated():
        # 
        # early stopping check
        # 
        global central_agent_process_index, when_all_processes_are_updated, prev_total_number_of_episodes, number_of_timesteps, number_of_episodes, number_of_updates,  process_index_to_temp_filter,  filtered_count,  act_val,  visits,  filtered_agents, episode_reward_trend, median_episode_rewards, episode_reward_trend
        total_number_of_episodes = number_of_episodes.value
        if total_number_of_episodes > config.early_stopping.min_number_of_episodes:
            output.number_of_episodes = total_number_of_episodes # keep the output value up to date
            
            # limiter
            if total_number_of_episodes >= prev_total_number_of_episodes + check_rate:
                prev_total_number_of_episodes = total_number_of_episodes
                # reset the rewards of filtered-out agents
                for process_index in range(config.number_of_processes):
                    if filtered_agents[process_index]:
                        median_episode_rewards[process_index] = 0
                per_episode_reward = sum(median_episode_rewards)/len(median_episode_rewards)
                episode_reward_trend.append(per_episode_reward)
                episode_reward_trend = output.episode_reward_trend = episode_reward_trend[-config.value_trend_lookback_size:]
                episode_reward_trend_value = trend_calculate(episode_reward_trend) / check_rate
                if len(episode_reward_trend) >= config.value_trend_lookback_size:
                    absolute_changes = [ abs(each) for each in utils.sequential_value_changes(episode_reward_trend)  ]
                    biggest_recent_change = max(absolute_changes)
                    if biggest_recent_change < config.early_stopping.lowerbound_for_max_recent_change:
                        print(f"Hit early stopping because biggest_recent_change: {biggest_recent_change} < {config.early_stopping.lowerbound_for_max_recent_change}")
                        stop_event.set()
                
                print(f'''{{"total_number_of_episodes":{total_number_of_episodes}, "number_of_timesteps":{number_of_timesteps.value}, "per_episode_reward":{per_episode_reward:.2f}, "episode_reward_trend_value": {episode_reward_trend_value},}},''')
                for each_step, each_min_value in config.early_stopping.thresholds.items():
                    # if meets the increment-based threshold
                    if total_number_of_episodes > each_step:
                        # enforce that it return the minimum 
                        if per_episode_reward < each_min_value:
                            print(f"Hit early stopping because per_episode_reward: {per_episode_reward} < {each_min_value}")
                            stop_event.set()
        
        # print("[starting when_all_processes_are_updated()]")
        all_malicious_actors_found = filtered_count.value == num_agents_byz
        if all_malicious_actors_found:
            return

        # Update values
        if number_of_updates.value != 0:
            # Compute gradient mean
            ucb_reward = ucb.reward_func(ucb.smart_gradient_of_agents)
            
            # Update Q-values
            ucb.update_step(ucb_reward)
            
            # 
            # Permanently disable an agent
            # 
            prev_amount_filtered = filtered_count.value
            for process_index, visit_count in enumerate(visits):
                if visit_count >= env_config.permaban_threshold:
                    filtered_agents[process_index] = 1
            filtered_count.value = sum(filtered_agents)
            
            # 
            # if someone was just recently permabaned
            # 
            if prev_amount_filtered < filtered_count.value:
                # Reset non-disabled visits/Q-values
                for process_index, visit_count in enumerate(visits):
                    if visit_count < env_config.permaban_threshold:
                        visits[process_index] = 0
                        ucb.value_per_process[process_index] = 0

        # Select next action
        process_index = ucb.choose_action()

        visits[process_index] += 1
        # Tell the multi processing barriers about the action
        process_index_to_temp_filter.value = process_index
        
        number_of_updates.value += 1
    
    all_updated_barrier = mp.Barrier(processes, lambda : None)

    def sync_updates():
        if filtered_count.value != num_agents_byz:
            num_updates = processes - (filtered_count.value + 1)
        else:
            num_updates = processes - filtered_count.value
        agent.average_updates(num_updates)
        agent.optimizer.step()
    update_barrier = mp.Barrier(processes, sync_updates)

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
            n_episodes=eval_n_episodes,
            eval_interval=eval_interval,
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
        random_seeds = np.arange(processes)

    def run_func(process_idx):
        config.verbose and print("[starting run_func()]")
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

        def f():
            train_loop(
                process_idx=process_idx,
                number_of_episodes=number_of_episodes,
                agent=local_agent,
                env=env,
                max_number_of_episodes=steps,
                outdir=outdir,
                max_episode_len=max_episode_len,
                evaluator=evaluator,
                successful_score=successful_score,
                stop_event=stop_event,
                exception_event=exception_event,
                eval_env=eval_env,
                global_step_hooks=global_step_hooks,
                logger=logger,
                all_updated_barrier=all_updated_barrier,
                update_barrier=update_barrier,
                process_index_to_temp_filter=process_index_to_temp_filter,
                filtered_agents=filtered_agents,
                byzantine_agent_number=num_agents_byz,
                median_episode_rewards=median_episode_rewards,
            )
        
        if profile:
            import cProfile

            cProfile.runctx(
                "f()", globals(), locals(), f"profile-{os.getpid()}.out"
            )
        else:
            try:
                f()
            except Exception as error:
                print(error)
                print()
                print()
                print()
                raise

        env.close()
        if eval_env is not env:
            eval_env.close()
    
    config.verbose and print("[about to call async_.run_async()]")
    async_.run_async(processes, run_func)
    config.verbose and print("[done calling async_.run_async()]")
    
    # make sure the median_episode_rewards is valid before stopping
    for process_index in range(config.number_of_processes):
        if filtered_agents[process_index]:
            median_episode_rewards[process_index] = 0
    stop_event.set()

    if evaluator is not None and use_tensorboard:
        evaluator.join_tensorboard_writer()

    return agent

def mp_to_numpy(mp_arr):
    return np.asarray(list(mp_arr))

def singleton(a_class):
    return a_class()