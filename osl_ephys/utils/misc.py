"""Miscellaneous utility classes and functions.

"""

import logging
import random
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)


def ensure_dir(dirname):
    """Create ``dirname`` (and parents) if it does not already exist.

    A no-op if the directory is already there. ``exist_ok=True`` avoids the
    race in parallel processing where two workers create the same folder.

    Parameters
    ----------
    dirname : str | pathlib.Path
        Directory to create.

    Returns
    -------
    created : bool
        True if the directory did not exist before this call, else False.
    """
    if isinstance(dirname, str):
        dirname = Path(dirname)
    created = not dirname.is_dir()
    dirname.mkdir(parents=True, exist_ok=True)
    return created


def proc_userargs(userargs, default, strict=True):
    """Merge ``userargs`` over a wrapper's ``default`` argument dict.

    Used by the preprocessing wrappers to validate and fill in their
    ``(dataset, userargs)`` arguments against a canonical default set.

    Parameters
    ----------
    userargs : dict | None
        The user-supplied arguments (``None`` is treated as ``{}``).
    default : dict
        The canonical key set with default values.
    strict : bool
        If True (default), raise ``KeyError`` on any user key not in
        ``default`` -- catches typos like ``{'ferqs': ...}`` for ``freqs``.

    Returns
    -------
    dict
        ``default`` updated with ``userargs``.
    """
    out = dict(default)
    for key, value in (userargs or {}).items():
        if strict and key not in default:
            raise KeyError(
                f"Key {key!r} is not expected in userargs; "
                f"accepted keys are {sorted(default)}.")
        out[key] = value
    return out


def mean_psd_in_band(psd_row, freqs, center, half_width):
    """Mean of a 1-D PSD over ``[center - half_width, center + half_width]``.

    Unlike :func:`osl_ephys.preprocessing.semp.metric.psd_band_stat` (which
    reduces an ``mne`` ``Spectrum`` over a ``[fmin, fmax]`` band), this works
    on a plain **numpy PSD row + its frequency axis**, takes the band as a
    **centre +/- half-width**, and **falls back to the single nearest bin**
    when the window is narrower than the frequency resolution (no bin lands
    inside it). That narrow-band fallback is why ``slice_reject`` uses this to
    score the tight slice-timing harmonics, rather than ``psd_band_stat``.

    Parameters
    ----------
    psd_row : numpy.ndarray
        1-D power spectral density (one channel/component).
    freqs : numpy.ndarray
        Frequency axis matching ``psd_row``.
    center : float
        Band centre frequency (Hz).
    half_width : float
        Half the band width (Hz).

    Returns
    -------
    float
        Mean density across the in-band bins, or the nearest-bin value.
    """
    mask = (freqs >= (center - half_width)) & (freqs <= (center + half_width))
    if not mask.any():
        return psd_row[np.argmin(np.abs(freqs - center))]   # nearest bin
    return psd_row[mask].mean()   # mean of density across bins


def set_random_seed(seed=None):
    """Set all random seeds.

    This includes Python's random module and NumPy.

    Parameters
    ----------
    seed : int
        Random seed.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    logger.info(f"Setting random seed to {seed}")

    random.seed(seed)
    np.random.seed(seed)
    return seed