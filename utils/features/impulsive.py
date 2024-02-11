import numpy as np

from utils.features.statistical import rms


def peak(input_array: np.array) -> float:
    """Calculates the maximum absolute value of an input array. Used to compute the other impulse
    metrics.

    Parameters:
        input_array (np.array): The input array.

    Returns:
        float: The peak value.
    """
    return np.max(np.abs(input_array))


def impulse_factor(input_array: np.ndarray) -> float:
    """Calculates the impulse factor of an input array.

    Compares the height of a peak to the mean level of the signal.

    Parameters:
        input_array (np.ndarray): The input array.

    Returns:
        float: The impulse factor.
    """
    peak_value = peak(input_array)
    abs_value_mean = np.mean(np.abs(input_array))
    imp = peak_value / abs_value_mean if abs_value_mean != 0 else 0.0

    return imp


def crest_factor(input_array: np.ndarray) -> float:
    """Calculates the crest factor of an input array.

    Faults often first manifest themselves in changes in
    the peakiness of a signal before they manifest in the energy
    represented by the signal root mean squared.

    The crest factor can provide an early warning for faults
    when they first develop.

    Parameters:
        input_array (np.ndarray): The input array.

    Returns:
        float: The crest factor of the input array.
    """
    peak_value = peak(input_array)
    rms_value = rms(input_array)
    crest_factor = peak_value / rms_value if rms_value != 0 else 0.0

    return crest_factor


def clearance_factor(input_array: np.ndarray) -> float:
    """Calculates the clearance factor of an input array.

    For rotating machinery, this feature is maximum for healthy bearings
    and goes on decreasing for defective ball, defective outer race, and
    defective inner race respectively.

    The clearance factor has the highest separation ability for defective inner race faults.

    Ref.: https://www.mathworks.com/help/predmaint/ug/signal-features.html

    Parameters:
        input_array (np.ndarray): The input array.

    Returns:
        float: The clearance factor.
    """
    peak_value = peak(input_array)
    denominator = np.mean(np.sqrt(np.abs(input_array))) ** 2
    clearance_factor = peak_value / denominator if denominator != 0 else 0.0

    return clearance_factor
