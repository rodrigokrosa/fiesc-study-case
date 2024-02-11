from typing import Callable, List

import numpy as np
from scipy import signal

from utils.signal_representations.fft import get_fft, get_fftfreq


def envelope_spectrum(input_array):
    """Calculate the envelope spectrum of an input array.

    Parameters:
        input_array (array-like): The input array containing the signal.

    Returns:
        array-like: The amplitudes of the envelope spectrum.
    """
    temporal_envelope = np.abs(signal.hilbert(input_array))
    amplitudes = get_fft(temporal_envelope)

    return amplitudes


def filter_envelope(
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
    # Get envelope through hilbert transform
    envelope_time_domain = np.abs(signal.hilbert(input_array))

    # Get envelope spectrum
    frequencies = get_fftfreq(len(envelope_time_domain), fs)
    spectrum_array = get_fft(envelope_time_domain)

    # Filter for the specified cutoff frequencies
    mask = (frequencies > cutoff[0]) & (frequencies < cutoff[1])
    filtered_envelope_spectrum = spectrum_array * mask

    # Apply transformation
    return method(filtered_envelope_spectrum)
