import numpy as np


def acc_to_vel(input_array: np.array, fs: int, convert: bool = False):
    """Converts acceleration to velocity.

    Parameters:
        input_array (np.array): Array of acceleration values.
        fs (int): Sampling frequency in Hz.
        convert (bool, optional): Whether to convert the input_array from g to mm/s^2. Defaults to False.

    Returns:
        np.array: Array of velocity values.
    """
    if convert:
        convert = 9.81 / 0.001  # g to mm/s^2
        input_array = convert * input_array

    zero_mean_input_array = input_array - input_array.mean()
    integral = zero_mean_input_array.cumsum()
    velocity = integral / fs
    velocity_zero_mean = velocity - velocity.mean()

    return velocity_zero_mean
