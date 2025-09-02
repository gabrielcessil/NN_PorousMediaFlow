import numpy as np

def Filter_outlier(array, lower_bound=None, upper_bound=None):
    mean = np.mean(array)
    std = np.std(array)
    if lower_bound is None: lower_bound = mean - 4 * std
    if upper_bound is None: upper_bound = mean + 4 * std
    return np.clip(array, lower_bound, upper_bound)

def Set_solids_to_value(output_array, input_array, value=0, solid_value=0):
    output_array[input_array==solid_value] = value
    return output_array
