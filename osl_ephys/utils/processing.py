"""Time series helpers used by sign flipping."""

from __future__ import annotations

import numpy as np


def standardize(x: np.ndarray, axis: int = 0, create_copy: bool = True) -> np.ndarray:
    """Standardizes a time series.

    Returns a time series with zero mean and unit variance.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    axis : int, optional
        Axis on which to perform the transformation.
    create_copy : bool, optional
        Should we return a new array containing the standardized data or modify
        the original time series array?

    Returns
    -------
    X :  np.ndarray
        Standardized data.
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    if create_copy:
        return (np.copy(x) - mean) / std
    return (x - mean) / std


def time_embed(x: np.ndarray, n_embeddings: int) -> np.ndarray:
    """Performs time embedding.

    Parameters
    ----------
    x : np.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    n_embeddings : int
        Number of samples in which to shift the data.

    Returns
    -------
    X : sliding_window_view
        Time embedded data. Shape is (n_samples - n_embeddings + 1,
        n_channels * n_embeddings).
    """
    if n_embeddings % 2 == 0:
        raise ValueError("n_embeddings must be an odd number.")

    # Shape of time embedded data
    te_shape = (x.shape[0] - (n_embeddings - 1), x.shape[1] * n_embeddings)

    # Perform time embedding
    X = (
        np.lib.stride_tricks.sliding_window_view(x=x, window_shape=te_shape[0], axis=0)
        .T[..., ::-1]
        .reshape(te_shape)
    )

    return X
