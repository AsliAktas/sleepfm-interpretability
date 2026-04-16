"""
Utility functions for SleepFM Interpretability Project.

Provides seed management, timing, normalization, and common helpers.
"""

import numpy as np
import random
import time
from functools import wraps
from typing import Optional


def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds for full reproducibility.

    Sets seeds for: numpy, python random, and torch (if available).
    Call this at the beginning of every script and test.

    Args:
        seed: Integer seed value.

    Notes:
        - Attempts to import torch; if not available, skips torch seeding
        - If torch is available, also sets:
          - torch.manual_seed(seed)
          - torch.cuda.manual_seed_all(seed)
          - torch.backends.cudnn.deterministic = True
          - torch.backends.cudnn.benchmark = False
    """
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def timer(func):
    """Decorator that prints execution time of a function.

    Usage:
        @timer
        def my_function():
            ...
        # Prints: "my_function took 3.45 seconds"

    Returns:
        Wrapped function that prints timing after execution.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper


def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row of a matrix to unit length.

    Args:
        vectors: np.ndarray of shape (n, d).

    Returns:
        Normalized array of same shape. Each row has L2 norm = 1.

    Notes:
        - Handles zero vectors gracefully: returns zero vector
          (does not produce NaN or Inf)
        - Uses np.linalg.norm with axis=1 and keepdims
    """
    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise ValueError("vectors must be a 2D numpy array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vectors / norms


def validate_noise_config(
    noise_config: dict,
    expected_modalities: tuple = ("eeg", "ecg", "resp", "emg"),
) -> dict:
    """Validate and fill noise_config dictionary.

    Ensures all expected modality keys are present.
    Missing keys default to 0.0. Extra keys raise ValueError.
    Negative values raise ValueError.

    Args:
        noise_config: Dictionary of modality -> noise level.
        expected_modalities: Tuple of expected modality names.

    Returns:
        Validated noise_config with all keys present.

    Raises:
        ValueError: If unknown modality key or negative noise value.
    """
    for key in noise_config:
        if key not in expected_modalities:
            raise ValueError(f"Unknown modality key: '{key}'")
        if not isinstance(noise_config[key], (int, float)):
            raise ValueError(f"Noise value for '{key}' must be numeric, got {type(noise_config[key]).__name__}")
        if noise_config[key] < 0:
            raise ValueError(f"Negative noise value for '{key}': {noise_config[key]}")
    result = {modality: 0.0 for modality in expected_modalities}
    result.update(noise_config)
    return result
