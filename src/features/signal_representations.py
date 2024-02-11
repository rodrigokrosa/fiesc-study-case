from typing import Dict, List

import pandas as pd

from utils.filters import array_filter
from utils.signal_representations import (
    acc_to_displ,
    acc_to_vel,
    envelope_spectrum,
    get_fft,
)


def get_signal_representations(
    dataset: pd.DataFrame, fs: int, signal_representations: Dict[str, List[str]]
) -> pd.DataFrame:
    """Generate signal representations for a given dataset.

    Args:
        dataset (pd.DataFrame): The input dataset.
        fs (int): The sampling frequency of the signals.
        signal_representations (Dict[str, List[str]]): A dictionary specifying the signal representations to generate.

    Returns:
        pd.DataFrame: The dataset with the generated signal representations.
    """
    # Velocity signal representation
    dataset[signal_representations["velocity"]] = dataset[
        signal_representations["acceleration"][:-1]
    ].map(lambda x: acc_to_vel(x, fs))

    # Displacement signal representation
    dataset[signal_representations["displacement"]] = dataset[
        signal_representations["acceleration"][:-1]
    ].map(lambda x: acc_to_displ(x, fs))

    # Spectrum signal representation
    dataset[signal_representations["fft"]] = dataset[signal_representations["acceleration"]].map(
        lambda x: get_fft(x)
    )

    # Envelope spectrum signal representation
    intermediate_acc_columns = [f"sensor_{num}/int_acceleration" for num in [1, 2, 3]]
    dataset[intermediate_acc_columns] = dataset[signal_representations["acceleration"][:-1]].map(
        lambda x: array_filter(x, fs, cutoff=[1000, 2500], filt_order=12)
    )
    dataset[signal_representations["envelope_spectrum"]] = dataset[intermediate_acc_columns].map(
        lambda x: envelope_spectrum(x)
    )
    dataset = dataset.drop(columns=intermediate_acc_columns)

    return dataset
