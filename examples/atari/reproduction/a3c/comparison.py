import logging
import sys
import os
import itertools

import optuna
from blissful_basics import FS, merge
from super_map import LazyDict

from main.config import config, env_config, info, args as cli_args
from train_a3c import args_from_config
import train_a3c

runs_for_comparison = 5
aspects_to_compare = {
    "profile": [
        "CARTPOLE",
        "LUNAR_LANDER",
        # "CHEETAH",
    ],
    "attack_method": [
        'sign',
        'act',
        # 'noise',
        "none"
    ],
    "defense_method": [ "softmax", "ucb", "none", ],
}

if __name__ == '__main__':
    from examples.atari.reproduction.a3c.comparison import main
    main()

def main():
    all_combinations = [ LazyDict(zip(aspects_to_compare.keys(), each)) for each in list(itertools.product(*aspects_to_compare.values())) ]
    for each in all_combinations:
        log_file_path = f"{info.path_to.comparison_log_folder}/{each.profile}__atk={each.attack_method}__def={each.defense_method}.log".lower()
        if FS.exists(log_file_path):
            print(log_file_path, " already done")
            continue
        
        # 
        # set env
        # 
        # grab all the data from the profile
        merge(
            old_value=config,
            new_value=info.as_dict["(project)"]["(profiles)"]["(default)"],
        )
        merge(
            old_value=config,
            new_value=info.as_dict["(project)"]["(profiles)"][each.profile],
        )
        
        # 
        # set attack_method
        # 
        if each.attack_method == "none":
            config.number_of_processes = 7
            config.number_of_malicious_processes = 0
        else:
            config.attack_method = each.attack_method
        
        # 
        # set defense_method
        # 
        if each.defense_method == "none":
            config.expected_number_of_malicious_processes = 0
        else:
            config.defense_method = each.defense_method
            config.expected_number_of_malicious_processes = 3
        
        from statistics import mean as average
        fitness_values = []
        print(f'''starting: {log_file_path}''')
        with RedirectOutput(log_file_path):
            args = args_from_config() # put them into the format that train_a3c expects
            # record multiple runs
            fitness_values = [ float(train_a3c.outer_training_function(args)) for each in range(runs_for_comparison) ]
            import json
            print(json.dumps({ **each, "log_file_path": log_file_path, "score": max(fitness_values), "fitness_values": fitness_values }))
        
        print(f'''{log_file_path} fitness_value = {max(fitness_values)}''')

class RedirectOutput:
    def __init__(self, path_to_file=None):
        import sys
        from blissful_basics import FS
        
        self.path_to_file = path_to_file
        parent_folder = FS.parent_path(path_to_file)
        FS.ensure_is_folder(parent_folder)
        self.temp_file_path = f"{parent_folder}/__temp__.log"
        self.file = open(self.temp_file_path, "w")
        
        self.real_stdout = sys.stdout
        self.real_stderr = sys.stderr
        sys.stdout = self.file
        sys.stderr = self.file
    
    def __enter__(self):
        return self
    
    def __exit__(self, _, error, traceback):
        from blissful_basics import FS
        if error is not None:
            import traceback
            traceback.print_exc()
        
        sys.stdout = self.real_stdout
        sys.stderr = self.real_stderr
        self.file.close()
        if error is not None:
            FS.move(self.temp_file_path, to=FS.dirname(self.path_to_file), new_name=FS.basename(self.path_to_file+".error"))
        else:
            FS.move(self.temp_file_path, to=FS.dirname(self.path_to_file), new_name=FS.basename(self.path_to_file))