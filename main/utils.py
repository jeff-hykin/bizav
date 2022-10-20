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

def save_all_charts_to(path, overwrite=True):
    import requests
    import file_system_py as FS
    FS.clear_a_path_for(path, overwrite=overwrite)
    FS.write(
        data=requests.get(url='http://localhost:9900/').text,
        to=path,
    )