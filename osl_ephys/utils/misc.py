"""Miscellaneous utility classes and functions."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import importlib
import os
import warnings

from typing import Optional, Tuple

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
                f"accepted keys are {sorted(default)}."
            )
        out[key] = value
    return out


def mean_psd_in_band(psd_row, freqs, center, half_width):
    """Mean of a 1-D PSD over ``[center - half_width, center + half_width]``.

    Unlike :func:`osl_ephys.preproc.semp.metric.psd_band_stat` (which
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
        return psd_row[np.argmin(np.abs(freqs - center))]  # nearest bin
    return psd_row[mask].mean()  # mean of density across bins


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


def system_call(cmd: str, verbose: bool = True) -> None:
    """Run a shell command.

    Parameters
    ----------
    cmd : str
        Command to execute.
    verbose : bool, optional
        Print the command before executing.
    """
    if verbose:
        print(cmd)
    os.system(cmd)


def top_eig(M: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the top-k eigenvalues and eigenvectors of a symmetric matrix.

    This function uses either :code:`scipy.sparse.linalg.eigsh`
    (for large matrices and a smaller number of eigenvectors)
    or :code:`numpy.linalg.eigh` (for smaller matrices or full
    eigendecomposition).

    Parameters
    ----------
    M : np.ndarray
        Symmetric matrix of shape (n, n).
    k : int
        Number of top eigenvalues/eigenvectors to compute.

    Returns
    -------
    vals : np.ndarray
        Top-k eigenvalues. Shape is (k,).
    vecs : np.ndarray
        Corresponding eigenvectors as columns. Shape is (n, k).
    """
    n = M.shape[0]
    vals = None
    if n > 300 and k < n // 2:
        from scipy.sparse.linalg import eigsh, ArpackNoConvergence

        try:
            vals, vecs = eigsh(M, k=k, which="LM")
            idx = np.argsort(-vals)
            vals, vecs = vals[idx], vecs[:, idx]
        except (RuntimeError, ArpackNoConvergence):
            vals = None
    if vals is None:
        vals, vecs = np.linalg.eigh(M)
        if k < n:
            vals, vecs = vals[-k:], vecs[:, -k:]

    # Make the largest element of each eigenvector positive.
    # This is to make the result reproducible.
    largest = np.argmax(np.abs(vecs), axis=0)
    vecs = vecs * np.sign(vecs[largest, np.arange(vecs.shape[1])])
    return vals, vecs


def override_dict_defaults(
    default_dict: dict, override_dict: Optional[dict] = None
) -> dict:
    """Helper function to update default dictionary values with user values.

    Parameters
    ----------
    default_dict : dict
        Dictionary of default values.
    override_dict : dict, optional
        Dictionary of user values.

    Returns
    -------
    new_dict : dict
        default_dict with values replaced by user values.
    """
    if override_dict is None:
        override_dict = {}
    return {**default_dict, **override_dict}


def soft_import(package):
    """Try to import a package raising friendly error if not present."""
    try:
        module = importlib.import_module(package)
    except (ImportError, ModuleNotFoundError):
        msg = f"Package '{package}' is required for this "
        msg += "function to run but cannot be imported. "
        msg += "Please install it into your python environment to continue."
        raise ModuleNotFoundError(msg)

    return module


class LazyModule:
    """A module imported on first use, so that a package only some functions
    need (glmtools and sails, which only the GLM module uses) does not have
    to be installed to import osl-ephys, and does not slow its import down."""

    def __init__(self, name: str, hint: str):
        self._name = name
        self._hint = hint
        self._module = None

    def __getattr__(self, attr):
        if self._module is None:
            try:
                self._module = importlib.import_module(self._name)
            except ImportError as e:
                raise ModuleNotFoundError(
                    f"'{self._name}' is needed here but is not installed; "
                    f"install it with: {self._hint}"
                ) from e
        return getattr(self._module, attr)


def lazy_import(name: str, hint: str | None = None) -> LazyModule:
    """A module imported on first attribute access (see :class:`LazyModule`)."""
    return LazyModule(name, hint or f"pip install {name}")
