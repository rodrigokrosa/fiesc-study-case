import numpy as np
from scipy import stats


def rms(input_array: np.ndarray) -> float:
    """Calculates the root mean square (RMS) of an input array.

    Parameters:
        input_array (np.ndarray): The input array.

    Returns:
        float: The root mean square (RMS) of the input array.
    """
    rms = np.sqrt(np.mean(input_array**2))

    return rms


def kurtosis(input_array: np.ndarray) -> float:
    """Calculate the kurtosis of an input array.

    Higher-order statistics provide insight to system behavior
    through the fourth moment (kurtosis).

    Parameters:
        input_array (np.ndarray): The input array.

    Returns:
        float: The kurtosis value.
    """
    kurtosis = stats.kurtosis(input_array, fisher=False)

    return kurtosis if not np.isnan(kurtosis) else 0


def skewness(input_array: np.array) -> float:
    """Calculate the skewness of an input array.

    Higher-order statistics provide insight to system behavior
    through the third moment (skewness).

    Parameters:
        input_array (np.array): The input array.

    Returns:
        float: The skewness of the input array.
    """
    skewness = stats.skew(input_array)

    return skewness if not np.isnan(skewness) else 0


def shape_factor(input_array: np.ndarray) -> float:
    """Calculates the shape factor of an input array.

    Shape factor is dependent on the signal shape while
    being independent of the signal dimensions.

    Ref. https://www.mathworks.com/help/predmaint/ug/signal-features.html

    Parameters:
        input_array (np.ndarray): The input array.

    Returns:
        float: The shape factor of the input array.
    """
    rms_value = rms(input_array)
    abs_value_mean = np.mean(np.abs(input_array))
    shape_factor = rms_value / abs_value_mean if abs_value_mean != 0 else 0.0

    return shape_factor


def peak_to_peak(input_array: np.ndarray) -> float:
    """Calculate the peak-to-peak value of an input array.

    Parameters:
        input_array (np.ndarray): The input array.

    Returns:
        float: The peak-to-peak value of the input array.
    """
    peak_to_peak = input_array.max() - input_array.min()

    return peak_to_peak
