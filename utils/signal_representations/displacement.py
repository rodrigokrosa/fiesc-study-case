import numpy as np

from utils.signal_representations.velocity import acc_to_vel


def acc_to_displ(input_array: np.array, fs: int, convert_acc=True) -> np.array:
    """Convert acceleration to displacement.

    Args:
        input_array (np.array): Array of acceleration values.
        fs (int): Sampling frequency in Hz.
        convert_acc (bool, optional): Whether to convert acceleration to velocity before converting to displacement.
            Defaults to True.

    Returns:
        np.array: Array of displacement values.
    """
    return acc_to_vel(acc_to_vel(input_array, fs, convert=convert_acc), fs, convert=False)
