import autorootcwd  # noqa

from configs.feature_configs import IMPULSIVE, SPECTRAL_BANDS, STATISTICS
from src.data.load_data import load_raw_data
from src.features.signal_representations import get_signal_representations
from src.features.specialized_features import (
    get_frequency_domain_features,
    get_time_domain_features,
)
from utils.features.frequency_domain import filter_spectrum
from utils.signal_representations import filter_envelope


def preprocess():
    """Preprocesses the dataset by loading raw data, extracting signal representations, and
    calculating time and frequency domain features.

    Returns:
        dataset (pandas.DataFrame): Preprocessed dataset.
    """
    dataset = load_raw_data()

    dataset = get_signal_representations(dataset)

    dataset = get_time_domain_features(
        dataset, feature_dict=STATISTICS, representations=["acceleration", "velocity"]
    )

    dataset = get_time_domain_features(
        dataset, feature_dict=IMPULSIVE, representations=["acceleration", "velocity"]
    )

    dataset = get_frequency_domain_features(
        dataset,
        feature_dict=SPECTRAL_BANDS,
        cutoff_frequencies=[
            [5, 500],
            [500, 1000],
            [5, 1000],
            [500, 1500],
            [1000, 1500],
            [1500, 2000],
            [2000, 3000],
            [3000, 5000],
        ],
        representation="fft",
        filter_func=filter_spectrum,
    )

    dataset = get_frequency_domain_features(
        dataset,
        feature_dict=SPECTRAL_BANDS,
        cutoff_frequencies=[[5, 300], [250, 500], [450, 750], [700, 1050]],
        representation="envelope_spectrum",
        filter_func=filter_envelope,
    )

    return dataset


if __name__ == "__main__":
    dataset = preprocess()

    representations_dataset = dataset[
        [
            "classes",
            "sensor_1/acceleration",
            "sensor_2/acceleration",
            "sensor_3/acceleration",
            "sensor_5/acceleration",
            "sensor_1/velocity",
            "sensor_2/velocity",
            "sensor_3/velocity",
            "sensor_1/displacement",
            "sensor_2/displacement",
            "sensor_3/displacement",
            "sensor_1/envelope_spectrum",
            "sensor_2/envelope_spectrum",
            "sensor_3/envelope_spectrum",
            "sensor_1/fft",
            "sensor_2/fft",
            "sensor_3/fft",
            "sensor_5/fft",
        ]
    ]

    representations_dataset.to_parquet("data/processed/signal_representation_features.parquet")

    specialized_dataset = dataset.drop(
        columns=[
            "sensor_1/acceleration",
            "sensor_2/acceleration",
            "sensor_3/acceleration",
            "sensor_5/acceleration",
            "sensor_1/velocity",
            "sensor_2/velocity",
            "sensor_3/velocity",
            "sensor_1/displacement",
            "sensor_2/displacement",
            "sensor_3/displacement",
            "sensor_1/envelope_spectrum",
            "sensor_2/envelope_spectrum",
            "sensor_3/envelope_spectrum",
            "sensor_1/fft",
            "sensor_2/fft",
            "sensor_3/fft",
            "sensor_5/fft",
        ]
    )

    specialized_dataset.to_parquet("data/processed/specialized_features.parquet")
