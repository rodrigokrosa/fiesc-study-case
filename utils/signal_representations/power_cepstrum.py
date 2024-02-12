import numpy as np
from scipy.signal import detrend

from utils.signal_representations.fft import get_fft, get_fftfreq


def get_cepstrum(
    input_array: np.array,
    fs: int,
) -> np.array:
    """Compute the power cepstrum of an input array.

    Parameters:
        input_array (np.array): The input array.
        fs (int): The sampling frequency.

    Returns:
        np.array: The power cepstrum of the input array.
    """
    # Compute the spectrum
    X = get_fft(input_array, remove_mean=False)

    # Return the logarithm of the power spectrum
    log_X = get_logarithm_spectrum(X**2)

    # Return the spectrum of the logarithmic power spectrum
    cepstrum = np.abs(get_fft(log_X, remove_mean=False)) ** 2

    return cepstrum


def get_logarithm_spectrum(
    X: np.array,
    dB_scale: bool = True,
    remove_mean: bool = False,
    remove_trend: bool = False,
) -> np.array:
    """Compute the logarithm of the spectral amplitudes.

    Parameters:
        X (np.array): The input array representing the spectral amplitudes.
        dB_scale (bool, optional): Whether to scale the amplitudes in decibels. Defaults to True.
        remove_mean (bool, optional): Whether to remove the mean value of the logarithmic spectrum. Defaults to False.
        remove_trend (bool, optional): Whether to remove the linear trend of the logarithmic spectrum. Defaults to False.

    Returns:
        np.array: The logarithmic spectrum.
    """
    # Substitute zero values in the FFT spectrum to avoid infinite
    # values in the log operation
    MIN_VALUE = 1e-16
    X = np.where(X == 0, MIN_VALUE, X)
    # Compute the logarithm (base 10) of the spectral amplitudes
    if dB_scale:
        log_X = 10 * np.log10(X)
    else:
        log_X = np.log10(X)
    # Remove the mean value of the logarithmic spectrum
    if remove_mean:
        log_X = log_X - np.mean(log_X)
    # Remove the linear trend of the logarithmic spectrum
    if remove_trend:
        log_X = detrend(log_X)

    return log_X


def quefrequencies(samples: int, fs: int):
    """Calculate the quefrency values for a given number of samples and sampling frequency.

    Parameters:
        samples (int): The number of samples.
        fs (int): The sampling frequency.

    Returns:
        np.array: An array of quefrency values.
    """
    return get_fftfreq(samples // 2, (samples / 2) / (fs / 2))
