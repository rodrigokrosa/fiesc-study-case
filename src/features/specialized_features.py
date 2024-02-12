import logging
from typing import Callable, Dict, List

import pandas as pd
from hydra.utils import instantiate

log = logging.getLogger(__name__)


def get_time_domain_features(
    dataset: pd.DataFrame,
    feature_dict: Dict[str, Callable],
    representations: List[str],
    signal_representations: Dict[str, List[str]],
) -> pd.DataFrame:
    """Extracts time domain features from the dataset.

    Args:
        dataset (pd.DataFrame): The input dataset.
        feature_dict (Dict[str, Callable]): A dictionary mapping feature names to corresponding feature extraction methods.
        representations (List[str]): A list of representations to extract features from.
        signal_representations (Dict[str, List[str]]): A dictionary mapping representation names to corresponding signal names.

    Returns:
        pd.DataFrame: The dataset with extracted time domain features.
    """

    # Loop through each combination of representation and feature
    for representation in representations:
        # Generate dictionary feature columns for each representation
        feature_columns = {
            f"{feat}": [f"{signal}/{feat}" for signal in signal_representations[representation]]
            for feat in feature_dict.keys()
        }

        for feat, method in feature_dict.items():
            intermediate_df = pd.DataFrame()
            log.info(f"Processing {representation}/{feat}...")

            # Instantiate config method for feature calculation
            method = instantiate(method)

            # Generate intermediate dataframe calculating the features
            intermediate_df[feature_columns[feat]] = dataset[
                signal_representations[representation]
            ].map(lambda x: method(x))

            dataset = pd.concat([dataset, intermediate_df], axis=1)

    return dataset


def get_frequency_domain_features(
    dataset: pd.DataFrame,
    fs: int,
    feature_dict: Dict[str, Callable],
    cutoff_frequencies: List[int],
    representation: str,
    signal_representations: Dict[str, List[str]],
    filter_func: Callable,
) -> pd.DataFrame:
    """Calculate frequency domain features for a given dataset.

    Args:
        dataset (pd.DataFrame): The input dataset.
        fs (int): The sampling frequency.
        feature_dict (Dict[str, Callable]): A dictionary mapping feature names to corresponding calculation methods.
        cutoff_frequencies (List[int]): A list of cutoff frequencies for frequency bands.
        representation (str): The signal representation to use.
        signal_representations (Dict[str, List[str]]): A dictionary mapping signal representations to corresponding signals.
        filter_func (Callable): The filtering function to apply.

    Returns:
        pd.DataFrame: The dataset with frequency domain features added.
    """

    # Generate dictionary with all combinations of features to be calculated
    freq_band_features = {
        f"{feat}/{freq_band[0]}-{freq_band[1]}": [
            f"{signal}/{feat}/{freq_band[0]}-{freq_band[1]}"
            for signal in signal_representations[representation]
        ]
        for freq_band in cutoff_frequencies
        for feat in feature_dict.keys()
    }

    # Loop through each combination of feature and frequency band
    for feat, method in feature_dict.items():
        # Instantiate config method for feature calculation
        method = instantiate(method)

        for freq_band in cutoff_frequencies:
            intermediate_df = pd.DataFrame()
            log.info(f"Processing {representation}/{feat}/{freq_band[0]}-{freq_band[1]}...")

            # Use acceleration representation for FFT features because of its filter_function
            filter_representation = representation if representation != "fft" else "acceleration"

            # Generate intermediate dataframe calculating the features for each freq band
            intermediate_df[freq_band_features[f"{feat}/{freq_band[0]}-{freq_band[1]}"]] = dataset[
                signal_representations[filter_representation]
            ].map(lambda x: filter_func(x, fs=fs, cutoff=freq_band, method=method))

            dataset = pd.concat([dataset, intermediate_df], axis=1)

    return dataset
