import os
from main.config import config, info
import pybullet  # register PyBullet enviroments with open ai gym

mal_args = f''' 
    --processes {config.number_of_processes}
    --malicious {config.number_of_malicious_processes}
    --mal_type {config.attack_method}
''' #TODO check act impl and impl 3rd attack

env_config = config.env_config
env_args =f'''
    --env {env_config.env_name}
    --steps {env_config.number_of_timesteps}
    --lr {env_config.learning_rate}
    --beta {env_config.beta}
    --t-max {env_config.t_max}
    --activation {env_config.activation}
    --hidden_size {env_config.hidden_size}
    --permaban_threshold {env_config.permaban_threshold}
'''

for seed in config.random_seeds:
    os.system((f'python {info.absolute_path_to.ac3_start} --seed ' + str(seed) + mal_args + env_args).replace("\n", " "))
