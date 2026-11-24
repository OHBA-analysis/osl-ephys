"""Band power from spectra, for the parcellation QC power maps."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np


def _validate(
    array: np.ndarray,
    correct_dimensionality: int,
    allow_dimensions: int,
    error_message: str,
) -> np.ndarray:
    """Checks if the dimensionality of an array is correct.

    Parameters
    ----------
    array : np.ndarray
        Array to be checked.
    correct_dimensionality : int
        The desired number of dimensions in the array.
    allow_dimensions : int
        The number of dimensions that is acceptable for the passed array to
        have.
    error_message : str
        Message to print if the array is not valid.

    Returns
    -------
    array : np.ndarray
        Array with the correct dimensionality.
    """
    array = np.array(array)

    # Add dimensions to ensure array has the correct dimensionality
    for dimensionality in allow_dimensions:
        if array.ndim == dimensionality:
            for i in range(correct_dimensionality - dimensionality):
                array = array[np.newaxis, ...]

    # Check no other dimensionality has been passed
    if array.ndim != correct_dimensionality:
        raise ValueError(error_message)

    return array


def get_frequency_args_range(frequencies: np.ndarray, frequency_range) -> List[int]:
    """Get min/max indices of a range in a frequency axis.

    Parameters
    ----------
    frequencies : np.ndarray
        1D numpy array containing a Frequency axis.
    frequency_range : list of len 2
        Minimum and maximum frequency.

    Returns
    -------
    args_range : list of len 2
        Indices for minimum and maximum frequency.
    """
    f_min_arg = np.argwhere(frequencies >= frequency_range[0])[0, 0]
    f_max_arg = np.argwhere(frequencies <= frequency_range[1])[-1, 0]
    if f_max_arg <= f_min_arg:
        raise ValueError("Cannot select requested frequency range.")
    return [f_min_arg, f_max_arg]


def variance_from_spectra(
    frequencies: np.ndarray,
    power_spectra: np.ndarray,
    components: Optional[np.ndarray] = None,
    frequency_range: Optional[List[float]] = None,
    method: str = "mean",
) -> np.ndarray:
    """Calculates variance from power spectra.

    Parameters
    ----------
    frequencies : np.ndarray
        Frequency axis of the PSDs. Only used if :code:`frequency_range` is
        given. Shape must be (n_freq,).
    power_spectra : np.ndarray
        Power/cross spectra for each channel.
        Shape must be (n_channels, n_freq) or (n_modes, n_channels, n_freq)
        or (n_sessions, n_modes, n_channels, n_freq).
        Cross spectra with space (..., n_channels, n_channels, n_freq)
        can also be passed.
    components : np.ndarray, optional
        Spectral components. Shape must be (n_components, n_freq).
    frequency_range : list, optional
        Frequency range in Hz to integrate the PSD over. Default is full range.
    method : str
        Should take the sum of the PSD over the frequency range
        (:code:`method="sum"`), the integral of the PSD (:code:`"integral"`),
        or take the average value of the PSD (:code:`method="mean"`).

    Returns
    -------
    var : np.ndarray
        Variance over a frequency band for each component of each mode.
        Shape is (..., n_channels).

    Notes
    -----
    To get the correct scaling so that the output of this function reflects
    the variance of the original data, then :code:`method='integral'` should
    be used. Otherwise, the output of this function has an arbitrary scaling.
    """

    # Validation
    if power_spectra.ndim == 2:
        # PSDs were passed
        power_spectra = power_spectra[np.newaxis, np.newaxis, ...]
        n_sessions, n_modes, n_channels, n_freq = power_spectra.shape

    elif power_spectra.shape[-2] != power_spectra.shape[-3]:
        # PSDs were passed, check dimensionality
        error_message = (
            "A (n_channels, n_freq), (n_modes, n_channels, n_freq) or "
            "(n_sessions, n_modes, n_channels, n_freq) array must be passed."
        )
        power_spectra = _validate(
            power_spectra,
            correct_dimensionality=4,
            allow_dimensions=[2, 3],
            error_message=error_message,
        )
        n_sessions, n_modes, n_channels, n_freq = power_spectra.shape

    else:
        # Cross spectra were passed, check dimensionality
        error_message = (
            "A (n_channels, n_channels, n_freq), "
            "(n_modes, n_channels, n_channels, n_freq) or "
            "(n_sessions, n_modes, n_channels, n_channels, n_freq) "
            "array must be passed."
        )
        power_spectra = _validate(
            power_spectra,
            correct_dimensionality=5,
            allow_dimensions=[3, 4],
            error_message=error_message,
        )
        n_sessions, n_modes, n_channels, n_channels, n_freq = power_spectra.shape

    if components is not None and frequency_range is not None:
        raise ValueError(
            "Only one of the arguments components or frequency range can be passed."
        )

    if frequency_range is not None and frequencies is None:
        raise ValueError(
            "If frequency_range is passed, frequenices must also be passed."
        )

    if method not in ["mean", "sum", "integral"]:
        raise ValueError("method should be 'mean', 'sum' or 'integral'.")

    # Number of spectral components
    if components is None:
        n_components = 1
    else:
        n_components = components.shape[0]

    # Calculate power maps for each array
    var = []
    for i in range(n_sessions):
        # Get PSDs
        if power_spectra.shape[-2] == power_spectra.shape[-3]:
            # Cross-spectra densities were passed
            psd = power_spectra[i, :, range(n_channels), range(n_channels)]
            psd = np.swapaxes(psd, 0, 1)
        else:
            # Only the PSDs were passed
            psd = power_spectra[i]

        # Concatenate over modes
        psd = psd.reshape(-1, n_freq)
        psd = psd.real

        if components is not None:
            # Calculate PSD for each spectral component
            p = components @ psd.T
            for j in range(n_components):
                p[j] /= np.sum(components[j])

        else:
            # Integrate over the given frequency range
            if frequency_range is None:
                if method == "sum":
                    p = np.sum(psd, axis=-1)
                elif method == "integral":
                    df = frequencies[1] - frequencies[0]
                    p = np.sum(psd * df, axis=-1)
                else:
                    p = np.mean(psd, axis=-1)
            else:
                [min_arg, max_arg] = get_frequency_args_range(
                    frequencies, frequency_range
                )
                if method == "sum":
                    p = np.sum(psd[..., min_arg : max_arg + 1], axis=-1)
                elif method == "integral":
                    df = frequencies[1] - frequencies[0]
                    p = np.sum(psd[..., min_arg : max_arg + 1] * df, axis=-1)
                else:
                    p = np.mean(psd[..., min_arg : max_arg + 1], axis=-1)

        p = p.reshape(n_components, n_modes, n_channels)
        var.append(p)

    return np.squeeze(var)
