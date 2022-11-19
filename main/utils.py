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



import json
from os.path import join
from cool_cache import cache
from blissful_basics import print
from informative_iterator import ProgressBar

ProgressBar.seconds_per_print = 0.08

class LogReader:
    # for parsing messy logs and turning them into a CSV/dataframe structure
    def __init__(self, ):
        self.preprocessors = []
    
    def add_preprocessor(self, function=None):
        if function!= None:
            self.preprocessors.append(function)
            return self
        else:
            def decorator_name(function_being_wrapped):
                self.preprocessors.append(function_being_wrapped)
                return function_being_wrapped
            return decorator_name
    
    @cache(watch_filepaths=lambda *args, **kwargs: args[-1:], watch_attributes=lambda self: [ self.preprocessors, ])
    def read(self, filepath, disable_logging=True):
        from informative_iterator import ProgressBar
        number_of_elements_so_far = 0
        output = {"__source__":[]}
        with open(filepath,'r') as f:
            output_string = f.read()
            with print.indent:
                for each in self.preprocessors:
                    output_string = each(output_string)
            for _, each_line in ProgressBar(output_string.splitlines(), disable_logging=disable_logging, title="reading data"):
                if each_line.endswith(","):
                    each_line = each_line[:-1]
                try:
                    parsed_line = json.loads(each_line)
                    if isinstance(parsed_line, dict):
                        output_keys = set(output.keys())
                        new_keys = [ each for each in parsed_line.keys() if each not in output_keys ]
                        for each_new_key in new_keys:
                            output[each_new_key] = [None]*number_of_elements_so_far
                        # add a row
                        for each_key, each_list in output.items():
                            if each_key == "__source__":
                                each_list.append(filepath)
                            else:
                                each_list.append(parsed_line.get(each_key, None))
                        number_of_elements_so_far += 1
                except Exception as error:
                    pass
        return output

from super_map import LazyDict
def _standardize_key(key, length):
    # floats with no decimal place are treated as valid indicies
    if isinstance(key, float):
        if int(key) == key:
            key = int(key)
    # single keys
    if isinstance(key, int):
        # allow negative indexing
        if key < 0:
            key = length + key
    return key

from super_hash import super_hash
    
class LiquidLog:
    def __init__(self, frame={}):
        self.frame = LazyDict(frame)
    
    @property
    def column_names(self):
        return tuple(self.frame.keys())
    
    @property
    def columns(self):
        return tuple(self.frame.values())
    
    @property
    def rows(self):
        return tuple(self)
    
    def copy(self):
        return LiquidLog({ each_column_name: list(each_column) for each_column_name, each_column in self.frame.items() })
    
    def without_columns(self, *columns):
        return LiquidLog({ each_key:each_value for each_key, each_value in self.frame.items() if each_key not in columns })
    
    def __repr__(self):
        length = self.__len__()
        columns = "\n".join(
            [
                f"                {each}=[... {length}],"
                    for each in self.column_names 
            ]
        )
        return f'''
            LiquidLog(dict(\n{columns}
            ))'''.replace("\n            ", "\n")
    
    def __len__(self):
        if len(self.column_names) == 0:
            return 0
        else:
            return len(self.frame[self.column_names[0]])
    
    def add(self, *values):
        number_of_elements_so_far = self.__len__()
        output_keys = set(self.frame.keys())
        for each_value in values:
            if isinstance(each_value, dict):
                new_keys = [ each for each in each_value.keys() if each not in output_keys ]
                for each_new_key in new_keys:
                    self.frame[each_new_key] = [None]*number_of_elements_so_far
                # add a row
                for each_key, each_list in self.frame.items():
                    each_list.append(each_value.get(each_key, None))
                number_of_elements_so_far += 1
        
        return self
    
    def to_dict(self):
        return self.frame
    
    def __json__(self):
        return self.frame
    
    def __iter__(self):
        length = self.__len__()
        if length:
            columns = self.column_names
            for each_index in range(length):
                yield LazyDict({
                    each_column: self.frame[each_column][each_index]
                        for each_column in columns
                })
    
    def __getitem__(self, key):
        length = self.__len__()
        if isinstance(key, str):
            return self.frame[key]
        elif not isinstance(key, slice):
            key = _standardize_key(key, length)
            return LazyDict({
                each_column: self.frame[each_column][key]
                    for each_column in self.column_names
            })
        elif isinstance(key, slice):
            start, stop, stride = key.indices(length)
            # start and stop will always be positive
            return LiquidLog().add(*(self[index] for index in range(start, stop, stride)))
        else:
            raise Exception(f'''{key} doesnt seem to be valid as a column name, row number, or slice of rows. \n    columns={self.column_names}\n    length={self.__len__()}''')
        
    def __setitem__(self, key, value):
        length = self.__len__()
        if isinstance(key, str):
            value = list(value)
            if len(value) != length:
                raise Exception(f'''tried to assign a column ({key}) but there are {length} rows, and the new column only had {len(values)} values, which means there isn't one value per row. First few values are: {values[:10]}''')
            self.frame[key] = value
        elif not isinstance(key, slice):
            key = _standardize_key(key, length)
            for each_column in self.column_names:
                self.frame[each_column][key] = value.get(each_column, None)
        elif isinstance(key, slice):
            start, stop, stride = key.indices(length)
            # start and stop will always be positive
            iterator = iter(value)
            for value_index, frame_index in enumerate(range(start, stop, stride)):
                self[frame_index] = next(iterator)
        else:
            raise Exception(f'''{key} doesnt seem to be valid as a column name, row number, or slice of rows. \n    columns={self.column_names}\n    length={self.__len__()}''')
    
    def concat(self, *others):
        others = list(others)
        other = others.pop(0)
        existing = set(self.column_names)
        new = tuple( each for each in other.column_names if each not in existing )
        columns = self.column_names + new
        new_frame = {}
        self_length = self.__len__()
        other_length = other.__len__()
        for each_column in columns:
            new_frame[each_column] = self.frame.get(each_column, [None]*self_length) + other.frame.get(each_column, [None]*other_length)
        if len(others) == 0:
            return LiquidLog(new_frame)
        else:
            return LiquidLog(new_frame).concat(*others)
    
    def grouped_by(self, *column_names):
        groups = {}
        for each in self:
            if len(column_names) == 1:
                group_key = each[column_names[0]]
            else:
                group_key = tuple(each[column_name] for column_name in column_names)
            groups.setdefault(group_key, LiquidLog())
            groups[group_key].add(each)
        
        return LazyDict(groups)
    
    def only_keep_if(self, function):
        indicies_to_keep = []
        for index, each in enumerate(self):
            # weird syntax is so that 
            # function(each_row)
            # function(each_row, index)
            # are chosen based on the lambda input
            if function(*(each, index)[0:function.__code__.co_argcount]):
                indicies_to_keep.append(index)
        
        new = LiquidLog({ each_column_name: [None]*len(indicies_to_keep) for each_column_name in self.column_names })
        mapping = tuple(enumerate(indicies_to_keep))
        for each_column in new.column_names:
            original = self.frame[each_column]
            new_list = new.frame[each_column]
            for new_index, old_index in mapping:
                new_list[new_index] = original[old_index]
        return new
        
    def map(self, function): # handles expand-columns and returning an iterable handles expand-rows
        new = LiquidLog()
        for index, each in enumerate(self):
            # weird syntax is so that 
            # function(each_row)
            # function(each_row, index)
            # are chosen based on the lambda input
            try:
                result = function(*(each, index)[0:function.__code__.co_argcount])
            except Exception as error:
                print(f'''index, row = {index}, {each}''')
                raise error
            if isinstance(result, dict):
                new.add(result)
            else:
                try:
                    result = iter(result)
                except TypeError as error:
                    pass
                for each in result:
                    new.add(each)
        return new
    
    def sort_by(self, *columns, reverse=False): # TODO
        length = self.__len__()
        new = LiquidLog({ each_column_name: [None]*length for each_column_name in self.column_names })
        for column_to_sort_by in reversed(columns):
            values = enumerate(self.frame[column_to_sort_by])
            mapping = tuple(
                enumerate(
                    each_index
                        for each_index, each_value in sorted(values, key=lambda each: each[1], reverse=reverse)
                )
            )
            # I'm sure theres a much much faster way to do this (assuming the user is sorting by many columns)
            for each_column_name in self.column_names:
                new_column = new.frame[each_column_name]
                old_column = self.frame[each_column_name]
                for each_new_index, each_old_index in mapping:
                    new_column[each_new_index] = old_column[each_old_index]
        return new
    
    def split(self, *, is_starter=None, is_ender=None, is_divider=None):
        indicies_to_split_before = []
        function = is_starter or is_ender or is_divider
        slices = []
        prev_end = None
        for index, each_row in enumerate(self):
            # weird syntax is so that 
            # function(each_row)
            # function(each_row, index)
            # are chosen based on the lambda input
            if function(*(each_row, index)[0:function.__code__.co_argcount]):
                if is_starter:
                    if prev_end == None:
                        slices.append(slice(0, index))
                        prev_end = index
                    else:
                        slices.append(slice(prev_end, index))
                        prev_end = index
                elif is_ender:
                    if prev_end == None:
                        slices.append(slice(0, index+1))
                        prev_end = index+1
                    else:
                        slices.append(slice(prev_end, index+1))
                        prev_end = index+1
                else:
                    if prev_end == None:
                        if index == 0:
                            prev_end = 1
                        else:
                            slices.append(slice(0, index))
                        prev_end = index+1
                    else:
                        slices.append(slice(prev_end, index))
                        prev_end = index+1
        length = index + 1
        if prev_end == None:
            slices.append(slice(0, length))
        elif prev_end != length:
            slices.append(slice(prev_end, length))
        
        frames = [
            { each_column: None for each_column in self.column_names }
                for each in slices
        ]
        slices_and_frames = tuple(zip(slices, frames))
        for each_column_name in self.column_names:
            column = self.frame[each_column_name]
            for each_slice, each_new_frame in slices_and_frames:
                each_new_frame[each_column_name] = column[each_slice]
            
        return tuple(LiquidLog(each) for each in frames)
    
    def without_duplicates(self):
        values = set()
        def checker(row):
            hash_value = super_hash(row)
            if hash_value in values:
                return False
            else:
                values.add(hash_value)
                return True
        return self.only_keep_if(checker)




log_reader = LogReader()
@log_reader.add_preprocessor()
def hyperparam_logs(string):
    lines = []
    for _, line in ProgressBar(string.splitlines(), title="parsing hyperparams"):
        if "}. Best is trial" in line and "and parameters: {" in line:
            start = line.index("and parameters: {") + len("and parameters: {") - 1
            end = line.index("}. Best is trial") + 1
            line = line[start:end].replace("None", "null").replace("True", 'true').replace("False", 'false').replace("'", '"')
            print(f'''line = {line}''')
        lines.append(line)
    return '\n'.join(lines)

@log_reader.add_preprocessor()
def bandit_logs(string):
    lines = []
    for _, line in ProgressBar(string.splitlines(), title="parsing ucb output"):
        # Step 50 3 visits [1.0, 3.0, 1.0, 34.0, 6.0, 1.0, 4.0]  episode_count: 31 q_vals: [-11.111, -9.362, -11.111, -8.611, -9.259, -11.111, -9.539]
        if line.startswith("Step ") and "episode_count:" in lines and "q_vals:" in line:
            step          = line[len("Step"):line.index("visits")].split()[0]
            visits        = line[line.index("visits"):line.index("episode_count:")]
            episode_count = line[line.index("episode_count:"):line.index("q_vals:")]
            q_vals        = line[line.index("q_vals:")+len("q_vals:"):]
            line = f'{{"step":{step}, "visits": {visits}, "episode_count":{episode_count}, "q_vals": {q_vals}}}'
        lines.append(line)
    return '\n'.join(lines)

@log_reader.add_preprocessor()
def final_evals(string):
    lines = [
        '{"final_eval":false}'
    ]
    for _, line in ProgressBar(string.splitlines(), title="parsing final_evals"):
        # Step 50 3 visits [1.0, 3.0, 1.0, 34.0, 6.0, 1.0, 4.0]  episode_count: 31 q_vals: [-11.111, -9.362, -11.111, -8.611, -9.259, -11.111, -9.539]
        if line.startswith("final_eval: "):
            line = line[len("final_eval: "):].replace("None", "null").replace("True", 'true').replace("False", 'false').replace("'", '"').replace("}", ', "final_eval": true }')
        lines.append(line)
    return '\n'.join(lines)