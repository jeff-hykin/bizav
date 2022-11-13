from quik_config import find_and_load
from blissful_basics import FS
import json

info = find_and_load(
    "info.yaml",
    cd_to_filepath=True,
    parse_args=True,
    defaults_for_local_data=["CARTPOLE"], # defaults to CARTPOLE profile
)
config = info.config
env_config = config.env_config
args = info.unused_args

# 
# study count
# 
if not FS.is_file(info.path_to.study_counter_file):
    FS.write(data='{"count": 0}', path=info.path_to.study_counter_file)

study_info = json.loads(FS.read(info.path_to.study_counter_file))
study_number = study_info["count"]
# update the count
study_info["count"] += 1
FS.write(data=json.dumps(study_info), path=info.path_to.study_counter_file)