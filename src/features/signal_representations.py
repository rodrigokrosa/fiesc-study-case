import pandas as pd

from configs.feature_configs import SAMPLING_RATE, SIGNAL_REPRESENTATIONS
from utils.filters import array_filter
from utils.signal_representations import (
    acc_to_displ,
    acc_to_vel,
    envelope_spectrum,
    get_fft,
)


def get_signal_representations(dataset: pd.DataFrame) -> pd.DataFrame:
    """Calculate different signal representations based on the given dataset.

    Args:
        dataset (pd.DataFrame): The input dataset containing acceleration data.

    Returns:
        pd.DataFrame: The dataset with additional columns representing velocity, displacement,
        spectrum, and envelope spectrum signal representations.
    """
    # Velocity signal representation
    dataset[SIGNAL_REPRESENTATIONS["velocity"]] = dataset[
        SIGNAL_REPRESENTATIONS["acceleration"][:-1]
    ].map(lambda x: acc_to_vel(x, SAMPLING_RATE))

    # Displacement signal representation
    dataset[SIGNAL_REPRESENTATIONS["displacement"]] = dataset[
        SIGNAL_REPRESENTATIONS["acceleration"][:-1]
    ].map(lambda x: acc_to_displ(x, SAMPLING_RATE))

    # Spectrum signal representation
    dataset[SIGNAL_REPRESENTATIONS["fft"]] = dataset[SIGNAL_REPRESENTATIONS["acceleration"]].map(
        lambda x: get_fft(x)
    )

    # Envelope spectrum signal representation
    intermediate_acc_columns = [f"sensor_{num}/int_acceleration" for num in [1, 2, 3]]
    dataset[intermediate_acc_columns] = dataset[SIGNAL_REPRESENTATIONS["acceleration"][:-1]].map(
        lambda x: array_filter(x, SAMPLING_RATE, cutoff=[1000, 2500], filt_order=12)
    )
    dataset[SIGNAL_REPRESENTATIONS["envelope_spectrum"]] = dataset[intermediate_acc_columns].map(
        lambda x: envelope_spectrum(x)
    )
    dataset = dataset.drop(columns=intermediate_acc_columns)

    return dataset
