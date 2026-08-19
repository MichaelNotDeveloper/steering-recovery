"""Tools for reproducible activation-steering experiments."""

from steering_recovery.denoiser import ActivationDenoiser, DenoiserBundle
from steering_recovery.normalization import ActivationNormalizer
from steering_recovery.streaming_data import TeacherForcedActivationIterableDataset

__all__ = [
    "ActivationDenoiser",
    "ActivationNormalizer",
    "DenoiserBundle",
    "TeacherForcedActivationIterableDataset",
]
__version__ = "0.1.0"
