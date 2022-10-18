from quik_config import find_and_load

info = find_and_load(
    "info.yaml",
    cd_to_filepath=True,
    parse_args=True,
    defaults_for_local_data=["CARTPOLE"], # defaults to CARTPOLE profile
)
config = info.config
env_config = config.env_config