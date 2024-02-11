import numpy as np


def rms_spectral(input_array: np.ndarray) -> float:
    """Calculate the root mean square (RMS) of the spectral magnitude of an input array.

    Parameters:
        input_array (np.ndarray): The input array representing the spectral magnitude.

    Returns:
        float: The root mean square (RMS) of the spectral magnitude.
    """
    return np.sqrt(np.sum(np.abs(input_array) ** 2)) / np.sqrt(2)
