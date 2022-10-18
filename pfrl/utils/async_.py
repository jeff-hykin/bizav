import warnings

import torch.multiprocessing as mp


class AbnormalExitWarning(Warning):
    """Warning category for abnormal subprocess exit."""

    pass

def run_async(n_process, run_func, *args):
    """Run experiments asynchronously.

    Args:
      n_process (int): number of processes
      run_func: function that will be run in parallel
    """

    processes = []

    for process_idx in range(n_process):
        processes.append(mp.Process(target=run_func, args=(process_idx, *args)))

    for each_process in processes:
        each_process.start()

    for process_idx, each_process in enumerate(processes):
        each_process.join()
        if each_process.exitcode > 0:
            warnings.warn(
                "Process #{} (pid={}) exited with nonzero status {}".format(
                    process_idx, each_process.pid, each_process.exitcode
                ),
                category=AbnormalExitWarning,
            )
        elif each_process.exitcode < 0:
            warnings.warn(
                "Process #{} (pid={}) was terminated by signal {}".format(
                    process_idx, each_process.pid, -each_process.exitcode
                ),
                category=AbnormalExitWarning,
            )
