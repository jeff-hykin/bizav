import silver_spectacle as ss
import sys
import json
import math
import file_system_py as FS

path = sys.argv[1]
data = json.loads(FS.read(path))

only_show_top = 10
min_score = -700
show_legend = True
vertical_label = "Median Reward-per-Episode"
horizonal_label = "Episode #"

colors = ["#83ecc9", "#c3e88d", "#82aaff", "#89ddff", "#c792ea", "#e57eb3", "#fec355", "#f07178", "#f78c6c",]
def wrap_around_get(number, a_list):
    return a_list[((number % len(a_list)) + len(a_list)) % len(a_list)]

lines = []
for each_curve_index, each_curve in enumerate(data):
    values = [
        each_element["per_episode_reward"]
            for each_element in each_curve
    ]
    
    lines.append({
        "label": f'Trial {each_curve_index}',
        "fill": False,
        "score": max(values) if values else -math.inf,
        "tension": 0.1,
        "data": [ 
            dict(
                x=each_element["total_number_of_episodes"]/1.00000001, # compensating for glitch in Chartjs
                y=each_element["per_episode_reward"],
            )
                for each_element in each_curve
        ]
    })

lines = sorted(lines, key=lambda each: each["score"])
lines = lines[-only_show_top:]
print(*[ each["label"]+"\n" for each in lines ])
for line_index, line in enumerate(lines):
    line["backgroundColor"] = wrap_around_get(line_index, colors)
    line["borderColor"] = wrap_around_get(line_index, colors)
    line["color"] = wrap_around_get(line_index, colors)

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
                "display": show_legend,
            },
        },
        "pointRadius": 1, # the size of the dots
        "scales": {
            "y": {
                "min": min_score,
                "title": {
                    "display": True,
                    "text": vertical_label,
                },
            },
            "x": {
                "title": {
                    "display": True,
                    "text": horizonal_label,
                },
            },
        }
    }
})

import file_system_py as FS
save_all_charts_to(path+".charts.html")