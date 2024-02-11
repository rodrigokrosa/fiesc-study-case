from typing import Callable, Dict, List

import pandas as pd

from configs.feature_configs import SAMPLING_RATE, SIGNAL_REPRESENTATIONS


def get_time_domain_features(
    dataset: pd.DataFrame, feature_dict: Dict[str, Callable], representations: List[str]
):
    """Preprocesses the dataset by applying feature engineering methods to the specified
    representations.

    Args:
        dataset (pd.DataFrame): The input dataset.
        feature_dict (Dict[str, Callable]): A dictionary mapping feature names to feature engineering methods.
        representations (List[str]): A list of representations to be processed.

    Returns:
        pd.DataFrame: The preprocessed dataset.
    """
    for representation in representations:
        feature_columns = {
            f"{feat}": [f"{signal}/{feat}" for signal in SIGNAL_REPRESENTATIONS[representation]]
            for feat in feature_dict.keys()
        }

        for feat, method in feature_dict.items():
            intermediate_df = pd.DataFrame()
            print(f"Processing {representation}/{feat}...")
            intermediate_df[feature_columns[feat]] = dataset[
                SIGNAL_REPRESENTATIONS[representation]
            ].map(lambda x: method(x))

            dataset = pd.concat([dataset, intermediate_df], axis=1)

    return dataset


def get_frequency_domain_features(
    dataset: pd.DataFrame,
    feature_dict: Dict[str, Callable],
    cutoff_frequencies: List[int],
    representation: str,
    filter_func: Callable,
):
    freq_band_features = {
        f"{feat}/{freq_band[0]}-{freq_band[1]}": [
            f"{signal}/{feat}/{freq_band[0]}-{freq_band[1]}"
            for signal in SIGNAL_REPRESENTATIONS[representation]
        ]
        for freq_band in cutoff_frequencies
        for feat in feature_dict.keys()
    }

    for feat, method in feature_dict.items():
        for freq_band in cutoff_frequencies:
            intermediate_df = pd.DataFrame()
            print(f"Processing {representation}/{feat}/{freq_band[0]}-{freq_band[1]}...")

            filter_representation = representation if representation != "fft" else "acceleration"

            intermediate_df[freq_band_features[f"{feat}/{freq_band[0]}-{freq_band[1]}"]] = dataset[
                SIGNAL_REPRESENTATIONS[filter_representation]
            ].map(lambda x: filter_func(x, fs=SAMPLING_RATE, cutoff=freq_band, method=method))

            dataset = pd.concat([dataset, intermediate_df], axis=1)

    return dataset
