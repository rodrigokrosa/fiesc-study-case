import numpy as np
from scipy.fft import fft, fftfreq


def get_fft(
    data: np.array,
    fs: int,
    remove_mean: bool = True,
):
    """Compute the one-dimensional discrete Fourier Transform.

    Parameters:
        data (array-like): Input data.
        fs (int): Sampling frequency of the input data.
        remove_mean (bool, optional): Whether to remove the mean of the input data. Default is True.

    Returns:
        freq (np.array): The frequencies of the spectrum.
        mag_spectrum (np.array): The magnitude spectrum of the input data.
    """
    # Number of samples
    n = int(len(data))

    # Remove average signal
    if remove_mean:
        data = data - np.mean(data)

    # FFT
    spectrum = fft(data, n=n) / n
    freq = fftfreq(n, 1 / fs)

    # One-sided spectrum and frequencies
    mag_spectrum = np.abs(spectrum[: n // 2])
    mag_spectrum[1:] = mag_spectrum[1:] * 2
    freq = freq[: n // 2]

    return freq, mag_spectrum
