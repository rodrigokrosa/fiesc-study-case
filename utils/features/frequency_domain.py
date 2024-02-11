from typing import Callable, List

import numpy as np

from utils.signal_representations.fft import get_fft, get_fftfreq


def spectral_rms(input_array: np.ndarray) -> float:
    """Calculate the root mean square (RMS) of the spectral magnitude of an input array.

    Parameters:
        input_array (np.ndarray): The input array representing the spectral magnitude.

    Returns:
        float: The root mean square (RMS) of the spectral magnitude.
    """
    return np.sqrt(np.sum(np.abs(input_array) ** 2)) / np.sqrt(2)


def filter_spectrum(
    input_array: np.array, fs: int, cutoff: List[int], method: Callable
) -> np.array:
    """Filters the input array using the specified cutoff frequencies. Applies transformation after
    filtering.

    Parameters:
        input_array (np.array): The input acceleration array to be filtered.
        cutoff (List[int]): The cutoff frequencies for the filter.
        method (Callable): The method to be used after filtering.

    Returns:
        np.array: The filtered spectrum after transformation.
    """
    frequencies = get_fftfreq(len(input_array), fs)
    spectrum_array = get_fft(input_array)
    mask = (frequencies > cutoff[0]) & (frequencies < cutoff[1])
    filtered_spectrum = spectrum_array * mask

    return method(filtered_spectrum)
