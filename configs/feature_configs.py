import numpy as np

from utils.features.frequency_domain import spectral_rms
from utils.features.impulsive import (
    clearance_factor,
    crest_factor,
    impulse_factor,
    peak,
)
from utils.features.statistical import (
    kurtosis,
    peak_to_peak,
    rms,
    shape_factor,
    skewness,
)

SAMPLING_RATE = 10_000

SIGNAL_REPRESENTATIONS = {
    "acceleration": [f"sensor_{num}/acceleration" for num in [1, 2, 3, 5]],
    "velocity": [f"sensor_{num}/velocity" for num in [1, 2, 3]],
    "displacement": [f"sensor_{num}/displacement" for num in [1, 2, 3]],
    "fft": [f"sensor_{num}/fft" for num in [1, 2, 3, 5]],
    "envelope_spectrum": [f"sensor_{num}/envelope_spectrum" for num in [1, 2, 3]],
}

STATISTICS = {
    "mean": np.mean,
    "std": np.std,
    "max": np.max,
    "min": np.min,
    "median": np.median,
    "rms": rms,
    "kurtosis": kurtosis,
    "skewness": skewness,
    "shape_factor": shape_factor,
    "peak_to_peak": peak_to_peak,
}

IMPULSIVE = {
    "peak": peak,
    "impulse_factor": impulse_factor,
    "crest_factor": crest_factor,
    "clearance_factor": clearance_factor,
}


SPECTRAL_BANDS = {
    "rms": spectral_rms,
    "peak": peak,
}
