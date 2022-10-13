import logging
import os
import signal
import subprocess
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
from torch import nn

from pfrl.experiments.evaluator import AsyncEvaluator
from pfrl.utils import async_, random_seed


def kill_all():
    if os.name == "nt":
        # windows
        # taskkill with /T kill all the subprocess
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(os.getpid())])
    else:
        pgid = os.getpgrp()
        os.killpg(pgid, signal.SIGTERM)
        sys.exit(1)


def train_loop(
    process_idx,
    env,
    agent,
    steps,
    outdir,
    counter,
    episodes_counter,
    stop_event,
    exception_event,
    max_episode_len=None,
    evaluator=None,
    eval_env=None,
    successful_score=None,
    logger=None,
    global_step_hooks=[],
    barrier=None,
    update_barrier=None,
    filter_act=None,
    filtered_agents=None,
    byzantine_agent_number=0
):

    logger = logger or logging.getLogger(__name__)

    if eval_env is None:
        eval_env = env

    def save_model():
        if process_idx == 0:
            # Save the current model before being killed
            dirname = os.path.join(outdir, "{}_except".format(global_t))
            agent.save(dirname)
            logger.info("Saved the current model to %s", dirname)

    try:

        episode_r = 0
        global_t = 0
        local_t = 0
        global_episodes = 0
        obs = env.reset()
        episode_len = 0
        successful = False

        while True:

            # a_t
            a = agent.act(obs)
            # o_{t+1}, r_{t+1}
            obs, r, done, info = env.step(a)
            local_t += 1
            episode_r += r
            episode_len += 1
            reset = episode_len == max_episode_len or info.get("needs_reset", False)
            agent.observe(obs, r, done, reset)

            if agent.updated:
                barrier.wait()  # Wait for all agents to complete rollout, then run filter()
                if byzantine_agent_number > 0:
                    # If not current UCB action and not permanently filtered, include agent's gradient in global model
                    if filter_act.value != process_idx and filtered_agents[process_idx] == 0: agent.add_update()
                else:
                    agent.add_update()
                update_barrier.wait()   # Wait for all agents to contribute their gradients to global model, the sync_updates() to step it's optimizer
                agent.after_update()    # Each agent will download the global model after the optimizer steps

            if done or reset:# Get and increment the global counter
                with counter.get_lock():
                    counter.value += 1
                    global_t = counter.value
                for hook in global_step_hooks:
                    hook(env, agent, global_t)

                metric_line = str(global_t)+'; '+str(process_idx)+'; '+str(episode_r)+'; '
                stats = agent.get_statistics()
                for item in stats:
                    metric_line += str(item) + '; '
                if stats[0] != -1:
                    logger.info(metric_line[:-2])

                # Evaluate the current agent
                if evaluator is not None:
                    eval_score = evaluator.evaluate_if_necessary(
                        t=global_t, episodes=global_episodes, env=eval_env, agent=agent
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

                with episodes_counter.get_lock():
                    episodes_counter.value += 1
                    global_episodes = episodes_counter.value

                if global_t >= steps or stop_event.is_set():
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
        barrier.abort()

    except (Exception, KeyboardInterrupt):
        save_model()
        raise

    if global_t == steps:
        # Save the final model
        dirname = os.path.join(outdir, "{}_finish".format(steps))
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
    step_before_disable=1
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
    logger = logger or logging.getLogger(__name__)

    for hook in evaluation_hooks:
        if not hook.support_train_agent_async:
            raise ValueError("{} does not support train_agent_async().".format(hook))

    # Prevent numpy from using multiple threads
    os.environ["OMP_NUM_THREADS"] = "1"

    counter = mp.Value("l", 0)
    episodes_counter = mp.Value("l", 0)

    act_val = mp.Array("d", 10)     # Q-values
    visits = mp.Array("d", 10)      # number of visits
    agent_rewards = mp.Array("d", 10)   # Used for UCB with rewards
    filtered_agents = mp.Array("l", 10)     # Permanently filtered agent memory
    for i in range(len(act_val)): act_val[i] = 0
    for i in range(len(visits)): visits[i] = 0
    for i in range(len(agent_rewards)): agent_rewards[i] = 0
    for i in range(len(filtered_agents)): filtered_agents[i] = 0

    time = mp.Value("l", 0)     # Number of total rollouts completed
    filter_act = mp.Value("i", 0)   # UCB action 'broadcast'
    filtered_count = mp.Value("l", 0)   # Number of permanently filtered agents

    def filter():
        if filtered_count.value == num_agents_byz: return

        # Update values
        if time.value != 0:
            # Compute gradient mean
            all_grads = []
            for i in range(len(agent.local_models)):
                my_grad = []
                # If filtered, don't include in gradient mean
                if filter_act.value == i or filtered_agents[i] == 1: continue
                for param in agent.local_models[i].parameters():
                    if param.grad is not None:
                        grad_np = param.grad.detach().clone().numpy().flatten()
                    else:
                        grad_np = np.zeros(param.size(), dtype=np.float).flatten()
                    for j in range(len(grad_np)):
                        my_grad.append(grad_np[j])
                all_grads.append(np.asarray(my_grad))
            all_grads = np.vstack(all_grads)
            r = (1-np.mean(np.var(all_grads, axis=-1)))*100

            # Update Q-values
            act_val[filter_act.value] += (r-act_val[filter_act.value]) / visits[filter_act.value]

            # Permanently disable an agent
            end_index = -1
            for i in range(len(visits)):
                if visits[i] == step_before_disable: end_index = i
            if end_index >= 0:
                filtered_count.value += 1
                visits[end_index] = step_before_disable+1
                filtered_agents[end_index] = 1
                # Reset non-disabled visits/Q-values
                for i in range(len(visits)):
                    if visits[i] < step_before_disable:
                        visits[i] = 0
                        act_val[i] = 0

        # Debug output
        print('Step', time.value, filter_act.value)
        # Select next action
        np_visits = mp_to_numpy(visits)
        np_act_vals = mp_to_numpy(act_val)

        print(list(np.round(np_visits, 2)))
        print(list(np.round(np_act_vals, 3)))

        # Get the true UCB t value
        ucb_timesteps = np.sum(np_visits) - (step_before_disable+1) * filtered_count.value
        # Compute UCB policy values (Q-value + uncertainty)
        values = np_act_vals + np.sqrt((np.log(ucb_timesteps)) / np_visits)

        # Mask permanently filtered agents
        for i in range(len(filtered_agents)):
            if filtered_agents[i] == 1: act_val[i] = 0

        # Initial selection (visits 0) #TODO properly select at random
        if np.min(np_visits) < 1:
            act = np.argmin(np_visits)
        else:   # UCB argmax policy
            act = np.argmax(values)

        # Increment visit for selected action
        visits[act] += 1
        # Tell the multi processing barriers about the action
        filter_act.value = act

        time.value += 1
    barrier = mp.Barrier(processes, filter)

    def sync_updates():
        if filtered_count.value != num_agents_byz:
            num_updates = processes - (filtered_count.value + 1)
        else:
            num_updates = processes - filtered_count.value
        #print('Sync updates', num_updates)
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
                counter=counter,
                episodes_counter=episodes_counter,
                agent=local_agent,
                env=env,
                steps=steps,
                outdir=outdir,
                max_episode_len=max_episode_len,
                evaluator=evaluator,
                successful_score=successful_score,
                stop_event=stop_event,
                exception_event=exception_event,
                eval_env=eval_env,
                global_step_hooks=global_step_hooks,
                logger=logger,
                barrier=barrier,
                update_barrier=update_barrier,
                filter_act=filter_act,
                filtered_agents=filtered_agents,
                byzantine_agent_number=num_agents_byz
            )

        if profile:
            import cProfile

            cProfile.runctx(
                "f()", globals(), locals(), "profile-{}.out".format(os.getpid())
            )
        else:
            f()

        env.close()
        if eval_env is not env:
            eval_env.close()

    async_.run_async(processes, run_func)

    stop_event.set()

    if evaluator is not None and use_tensorboard:
        evaluator.join_tensorboard_writer()

    return agent

def mp_to_numpy(mp_arr):
    buff = []
    for i in range(len(mp_arr)):
        buff.append(mp_arr[i])
    return np.asarray(buff)
