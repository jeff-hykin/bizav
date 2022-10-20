def trend_calculate(values):
    if len(values) < 2:
        return 0
    
    # get pairwise elements
    changes = [ each - prev for prev, each in zip(values[0:-1], values[1:]) ]
    return sum(changes)/len(changes)
        