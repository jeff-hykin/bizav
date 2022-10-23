import silver_spectacle as ss
import sys
import json
import file_system_py as FS

path = sys.argv[1]
data = json.loads(FS.read(path))

lines = []
for each_curve_index, each_curve in enumerate(data):
    lines.append({
        "label": f'{each_curve_index}',
        "fill": True,
        "tension": 0.1,
        "data": [ 
            dict(
                x=each_element["total_number_of_episodes"],
                y=each_element["per_episode_reward"]
            )
                for each_element in each_curve
        ]
    })

def save_all_charts_to(path, overwrite=True):
    import requests
    import file_system_py as FS
    FS.clear_a_path_for(path, overwrite=overwrite)
    FS.write(
        data=requests.get(url='http://localhost:9900/').text,
        to=path,
    )

import silver_spectacle as ss
ss.DisplayCard("chartjs", {
    "type": 'line',
    "data": {
        "datasets": lines,
    },
    "options": {
        "plugins": {
            "legend": {
                "display": False,
            },
        },
        "pointRadius": 3, # the size of the dots
        "scales": {
            "y": {
                "min": -700,
                # "max": 200,
            },
        }
    }
})

import file_system_py as FS
save_all_charts_to(FS.local_path("./charts.html"))