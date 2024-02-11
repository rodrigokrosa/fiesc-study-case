import logging

import autorootcwd  # noqa
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.datamodule.components.load_data import load_raw_data
from src.features.signal_representations import get_signal_representations
from src.features.specialized_features import (
    get_frequency_domain_features,
    get_time_domain_features,
)

log = logging.getLogger(__name__)


@hydra.main(
    version_base="1.3",
    config_path="../configs/feature_engineering",
    config_name="feature_configs.yaml",
)
def preprocess(cfg: DictConfig) -> None:
    """Preprocesses the dataset by loading raw data, extracting signal representations, and
    calculating time and frequency domain features.

    Returns:
        dataset (pandas.DataFrame): Preprocessed dataset.
    """
    log.info("Starting preprocessing...")

    log.info("Loading raw data...")
    dataset = load_raw_data()

    log.info("Extracting signal representations...")
    dataset = get_signal_representations(
        dataset,
        fs=cfg.sampling_rate,
        signal_representations=OmegaConf.to_object(cfg.representations.signal_representations),
    )

    log.info("Calculating time domain statistical features...")
    dataset = get_time_domain_features(
        dataset,
        feature_dict=cfg.features.statistical,
        representations=["acceleration", "velocity"],
        signal_representations=OmegaConf.to_object(cfg.representations.signal_representations),
    )

    log.info("Calculating time domain impulsive features...")
    dataset = get_time_domain_features(
        dataset,
        feature_dict=cfg.features.impulsive,
        representations=["acceleration", "velocity"],
        signal_representations=OmegaConf.to_object(cfg.representations.signal_representations),
    )

    log.info("Calculating frequency domain spectral band features...")
    dataset = get_frequency_domain_features(
        dataset,
        fs=cfg.sampling_rate,
        feature_dict=cfg.features.spectral,
        cutoff_frequencies=OmegaConf.to_object(cfg.filters.spectral_bands),
        representation="fft",
        signal_representations=OmegaConf.to_object(cfg.representations.signal_representations),
        filter_func=instantiate(cfg.filters.spectrum),
    )

    log.info("Calculating frequency domain envelope spectrum features...")
    dataset = get_frequency_domain_features(
        dataset,
        fs=cfg.sampling_rate,
        feature_dict=cfg.features.spectral,
        cutoff_frequencies=OmegaConf.to_object(cfg.filters.envelope_bands),
        representation="envelope_spectrum",
        signal_representations=OmegaConf.to_object(cfg.representations.signal_representations),
        filter_func=instantiate(cfg.filters.envelope),
    )

    log.info("Splitting dataset into representations and specialized features...")
    representations_dataset = dataset[OmegaConf.to_object(cfg.representations.keep_cols)]
    specialized_dataset = dataset.drop(columns=OmegaConf.to_object(cfg.features.drop_cols))

    log.info("Saving datasets...")
    representations_dataset.to_parquet("data/processed/signal_representation_features.parquet")

    specialized_dataset.to_parquet("data/processed/specialized_features.parquet")
    log.info("Done.")


if __name__ == "__main__":
    preprocess()
