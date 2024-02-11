from typing import List

import numpy as np
from scipy import signal


def bandpass_filter(
    input_array: np.array,
    N: int,
    cutoff: List[int],
    fs: int,
    btype: str = "bandpass",
):
    """Apply a butterworth filter to the input array.

    Parameters:
        input_array (np.array): The input array to be filtered.
        N (int): The order of the filter.
        cutoff (List[int]): The cutoff frequencies of the filter.
        fs (int): The sampling frequency of the input array.
        btype (str, optional): The type of the filter. Defaults to "bandpass".
        analog (bool, optional): Whether the filter is analog or digital. Defaults to False.
        output (str, optional): The output format of the filter coefficients. Defaults to "sos".

    Returns:
        np.array: The filtered output array.
    """
    nyq = 0.5 * fs
    Wn = np.asarray(cutoff, dtype=np.float64) / nyq

    butter_coef = signal.butter(N, Wn, btype=btype, output="sos")

    return signal.sosfiltfilt(butter_coef, input_array, padlen=int(N / 2), padtype="even")


def array_filter(input_array: np.array, fs: int, cutoff: List[int], filt_order: int = 3):
    """Apply a bandpass filter to the input array.

    Parameters:
    - array (np.array): Input array to be filtered.
    - fs (int): Sampling frequency of the input array.
    - cutoff (List[int]): Frequency range for the bandpass filter.
    - filt_order (int): Order of the filter (default: 3).

    Returns:
    - filtered_array (np.array): Filtered array.
    """
    lowcut = float(cutoff[0])
    highcut = float(cutoff[1])
    nyq = fs * 0.5
    Wn_lowcut = lowcut / nyq
    Wn_highcut = highcut / nyq

    if (0 < Wn_lowcut < 1) and (0 < Wn_highcut < 1):
        filtered_array = bandpass_filter(
            input_array, filt_order, [lowcut, highcut], fs, btype="bandpass"
        )
    elif (0 < Wn_lowcut < 1) and (Wn_highcut >= 1):
        filtered_array = bandpass_filter(input_array, filt_order, lowcut, fs, btype="highpass")
    elif (Wn_lowcut <= 0) and (0 < Wn_highcut < 1):
        filtered_array = bandpass_filter(input_array, filt_order, highcut, fs, btype="lowpass")
    else:
        filtered_array = np.zeros(input_array.shape)

    return filtered_array
