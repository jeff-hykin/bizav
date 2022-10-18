import os
from quik_config import find_and_load

info = find_and_load(
    "info.yaml",
    cd_to_filepath=True,
    parse_args=True,
    defaults_for_local_data=["CARTPOLE"], # defaults to CARTPOLE profile
)
config = info.config


mal_args = f''' 
    --processes {config.number_of_processes}
    --malicious {config.number_of_malicious_processes}
    --mal_type {config.attack_method}
''' #TODO check act impl and impl 3rd attack
env_args =f'''
    --env {config.env_name}
    --steps {config.number_of_timesteps}
    --lr {config.learning_rate}
    --beta {config.beta}
    --t-max {config.t_max}
    --activation {config.activation}
    --hidden_size {config.hidden_size}
    --ucb_disable {config.ucb_disable}
'''

for seed in config.random_seeds:
    os.system((f'python {info.absolute_path_to.ac3_start} --seed ' + str(seed) + mal_args + env_args).replace("\n", " "))
