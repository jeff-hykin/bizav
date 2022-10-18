import os
import numpy as np
import sys
import shutil
import matplotlib.pyplot as plt

log_dir = '../../../../../../../Desktop'
log_folders = [folder for folder in next(os.walk(log_dir))[1]]
num_runs = 0
run_dict = dict()

for folder in log_folders:
    name = ''
    with open(log_dir + os.sep + folder + os.sep + 'args.txt') as fp:
        mal_val = 0
        for line in fp:
            line = line[1:-1].replace('"', '')
            for spl in line.split(', '):
                if 'malicious' in spl:
                    mal_val = float(spl.split(': ')[1])
                    
                if 'scalar' in spl or 'stat' in spl or 'malicious' in spl:
                    name += spl.split(': ')[1]
        #if mal_val != 0: continue

    if not name:
        name = 'base'
        print(folder)
        """with open(log_dir + os.sep + folder + os.sep + 'args.txt') as fp:
            for line in fp:
                print(line)
                print()"""
    #if '2' not in name and 'base' not in name: continue
    #print(folder)
    #shutil.rmtree(log_dir + os.sep + folder)
    #continue

    log_files = [folder for folder in next(os.walk(log_dir + os.sep + folder))[2]]
    for log_file in log_files:
        if log_file[-4:] == '.log': 
                log_path = log_dir + os.sep + folder + os.sep + log_file
            
                this_run_scores = []
                if name not in run_dict:
                    run_dict[name] = []
                run_dict[name].append(this_run_scores)

                with open(log_path) as fp:
                    for line in fp:
                        if 'Saved' in line: continue
                        this_run_scores.append(float(line[22:].split(';')[2].strip()))

import collections
window_size = 100
std_scale = 1.0
for run_name in run_dict.keys():
    print(run_name, len(run_dict[run_name]))
    list_runs = run_dict[run_name]
    min_len = min([len(run) for run in list_runs])
    list_runs = [run[:min_len] for run in list_runs]
    avg = np.mean(list_runs, axis=0)#/len(list_runs)
    err = np.std(list_runs, axis=0) * std_scale
    print(avg)

    avg_window = collections.deque(maxlen=window_size)
    avg_final = []
    for item in avg:
        avg_window.append(item)
        avg_final.append(np.mean(avg_window))
    avg_final = np.asarray(avg_final)
    print(avg_final)

    err_window = collections.deque(maxlen=window_size)
    err_final = []
    for item in err:
        err_window.append(item)
        err_final.append(np.mean(err_window))
    err_final = np.asarray(err_final)

    plt.title(run_name)
    #plt.ylim(0, 200)
    #plt.xlim(0,12000)
    plt.plot(avg_final, label=run_name)
    plt.fill_between([i for i, _ in enumerate(avg)], avg_final-err_final, avg_final+err_final, alpha=0.4)

plt.legend()
plt.show()
