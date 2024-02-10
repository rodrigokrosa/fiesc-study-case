import numpy as np
from scipy.fft import fft, fftfreq


def get_fft(
    data: np.array,
    remove_mean: bool = True,
) -> np.array:
    """Compute the one-dimensional discrete Fourier Transform.

    Parameters:
        data (array-like): Input data.
        remove_mean (bool, optional): Whether to remove the mean of the input data. Default is True.

    Returns:
        np.array: The magnitude spectrum of the input data.
    """
    # Number of samples
    n = int(len(data))

    # Remove average signal
    if remove_mean:
        data = data - np.mean(data)

    # FFT
    spectrum = fft(data, n=n, axis=0) / n

    # One-sided spectrum and frequencies
    mag_spectrum = np.abs(spectrum[: n // 2])
    mag_spectrum[1:] = mag_spectrum[1:] * 2

    return mag_spectrum


def get_fftfreq(samples: int, fs: int) -> np.array:
    """Calculate the frequencies of the FFT output.

    Args:
        samples (int): The number of samples in the input signal.
        fs (int): The sampling frequency of the input signal.

    Returns:
        np.array: An array of frequencies corresponding to the FFT output.
    """
    freq = fftfreq(samples, 1 / fs)
    freq = freq[: samples // 2]
    return freq
