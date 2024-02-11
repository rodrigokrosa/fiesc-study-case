import numpy as np
from scipy import signal

from utils.signal_representations.fft import get_fft


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
