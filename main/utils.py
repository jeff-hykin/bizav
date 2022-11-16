def sequential_value_changes(values):
    if len(values) < 2:
        return []
    # get pairwise elements
    return [ each - prev for prev, each in zip(values[0:-1], values[1:]) ]

def trend_calculate(values):
    if len(values) < 2:
        return 0
    
    # get pairwise elements
    changes = sequential_value_changes(values)
    return sum(changes)/len(changes)

def rolling_average(a_list, window):
    results = []
    if len(a_list) < window * 2:
        return a_list
    near_the_end = len(a_list) - 1 - window 
    for index, each in enumerate(a_list):
        # at the start
        if index < window:
            average_items = a_list[0:index]+a_list[index:index+window]
        # at the end
        elif index > near_the_end:
            average_items = a_list[index-window:index]+a_list[index:len(a_list)]
        else:
            # this could be done a lot more efficiently with a rolling sum, oh well! ¯\_(ツ)_/¯ 
            average_items = a_list[index-window:index+window+1]
        results.append(sum(average_items)/len(average_items))
    return results

def save_all_charts_to(path, overwrite=True):
    import requests
    import file_system_py as FS
    FS.clear_a_path_for(path, overwrite=overwrite)
    FS.write(
        data=requests.get(url='http://localhost:9900/').text,
        to=path,
    )

figures = {}
lines = {}
def plot_line(plot_name, line_name, new_x_values, new_y_values):
    import time
    from plotly import graph_objects
    import plotly.express as px
    from blissful_basics import FS
    
    go = graph_objects
    
    figure = figures.setdefault(
        plot_name,
        go.Figure(
            data=[],
            layout=go.Layout(
                title=go.layout.Title(text=plot_name)
            ),
        )
    )
    figure.update_layout(template="plotly_dark")
    lines.setdefault(plot_name, {})
    line_index = lines[plot_name].get(line_name, None)
    if line_index == None:
        line_index = len(figure.data)
        lines[plot_name][line_name] = line_index
        figure.add_scatter(x=[], y=[], mode='lines')
    
    scatter_plot = figure.data[line_index]
    scatter_plot.name = line_name
    if not isinstance(scatter_plot.x, tuple):
        scatter_plot.x = tuple(scatter_plot.x)
    if not isinstance(scatter_plot.y, tuple):
        scatter_plot.y = tuple(scatter_plot.y)
    
    # append new data
    scatter_plot.x = scatter_plot.x + tuple(new_x_values)
    scatter_plot.y = scatter_plot.y + tuple(new_y_values)
    
    # save
    save_path = f"figures/{plot_name}.html"
    FS.ensure_is_folder(FS.parent_path(save_path))
    figure.write_html(save_path)


def variance_plot():
    # NOTE: this is just a template for whenever I feel like taking the time to make this generic
    import plotly.graph_objects as go
    import numpy as np
    
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    x_rev = x[::-1]

    # Line 1
    y1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y1_upper = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    y1_lower = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    y1_lower = y1_lower[::-1]

    # Line 2
    y2 = [5, 2.5, 5, 7.5, 5, 2.5, 7.5, 4.5, 5.5, 5]
    y2_upper = [5.5, 3, 5.5, 8, 6, 3, 8, 5, 6, 5.5]
    y2_lower = [4.5, 2, 4.4, 7, 4, 2, 7, 4, 5, 4.75]
    y2_lower = y2_lower[::-1]

    # Line 3
    y3 = [10, 8, 6, 4, 2, 0, 2, 4, 2, 0]
    y3_upper = [11, 9, 7, 5, 3, 1, 3, 5, 3, 1]
    y3_lower = [9, 7, 5, 3, 1, -.5, 1, 3, 1, -1]
    y3_lower = y3_lower[::-1]


    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y1_upper+y1_lower,
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Fair',
    ))
    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y2_upper+y2_lower,
        fill='toself',
        fillcolor='rgba(0,176,246,0.2)',
        line_color='rgba(255,255,255,0)',
        name='Premium',
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y3_upper+y3_lower,
        fill='toself',
        fillcolor='rgba(231,107,243,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Ideal',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y1,
        line_color='rgb(0,100,80)',
        name='Fair',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y2,
        line_color='rgb(0,176,246)',
        name='Premium',
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y3,
        line_color='rgb(231,107,243)',
        name='Ideal',
    ))

    fig.update_traces(mode='lines')
    fig.show()