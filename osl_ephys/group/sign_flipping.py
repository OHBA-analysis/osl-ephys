"""Sign flipping.

Source reconstruction leaves the sign of each channel (parcel) arbitrary, so the
same parcel can have opposite polarity across sessions. Before pooling sessions
we align these signs: for each session we search for the ``+1``/``-1`` per-channel
vector whose (time-delay embedded, standardized) covariance best matches a
template.

The search is done by `osl-signflip <https://pypi.org/project/osl-signflip>`_,
which provides four methods (see :data:`METHODS`):

- ``"sdp"`` (default): semidefinite relaxation with Goemans-Williamson rounding.
- ``"spectral"``: spectral synchronization (leading eigenvector).
- ``"belief_propagation"``: loopy belief propagation.
- ``"random"``: random search.

The hyperparameters of each method are passed as keyword arguments, e.g.
``find_flips(cov, template_cov, 15, method="sdp", sdp_n_restarts=10)``; see the
osl-signflip documentation of ``find_flips_<method>`` for the options.

For the common case of sign flipping a single parcellated fif file (or
:class:`mne.io.Raw`) to match a saved template covariance, see
:func:`sign_flip_mne_raw`.
"""

import inspect
import logging
from typing import Optional, Tuple, Union

import mne
import numpy as np
import osl_signflip
from osl_signflip import (
    apply_flips_to_covariance,
    compute_covariance,
    covariance_matrix_correlation,
    find_template,
)

from osl_ephys.parcellation.parcellation import convert_to_mne_raw, save_as_fif

logger = logging.getLogger("osl_ephys")

METHODS = osl_signflip.METHODS  # names of the available sign flipping methods
DEFAULT_METHOD = "sdp"  # method used when none is given

# Arguments of the osl-signflip algorithms that are not method hyperparameters
_DATA_ARGS = ("cov", "template_cov", "n_embeddings")

__all__ = [
    "DEFAULT_METHOD",
    "METHODS",
    "apply_flips_to_covariance",
    "compute_covariance",
    "covariance_matrix_correlation",
    "find_flips",
    "find_template",
    "get_method_hyperparameters",
    "sign_flip",
    "sign_flip_mne_raw",
]


def get_method_hyperparameters(method: str) -> dict:
    """Hyperparameters of a sign flipping method and their default values.

    Parameters
    ----------
    method : str
        One of :data:`METHODS`.

    Returns
    -------
    hyperparameters : dict
        Maps each hyperparameter name to its default value.
    """
    if method not in METHODS:
        raise ValueError(
            f"Invalid sign flipping method {method!r}. "
            f"Must be one of {', '.join(map(repr, METHODS))}."
        )
    solver = getattr(osl_signflip, f"find_flips_{method}")
    return {
        name: p.default
        for name, p in inspect.signature(solver).parameters.items()
        if name not in _DATA_ARGS
    }


def find_flips(
    cov: np.ndarray,
    template_cov: np.ndarray,
    n_embeddings: int,
    method: str = DEFAULT_METHOD,
    **kwargs,
) -> Tuple[np.ndarray, float]:
    """Find the ``±1`` per-channel flips aligning ``cov`` to a template.

    Parameters
    ----------
    cov : np.ndarray
        This session's embedded covariance (from :func:`compute_covariance`).
        Shape is (n_channels * n_embeddings, n_channels * n_embeddings).
    template_cov : np.ndarray
        Template embedded covariance to align to (same shape as ``cov``).
    n_embeddings : int
        Number of time-delay embeddings used to build the covariances.
    method : str, optional
        Sign flipping method, one of :data:`METHODS`. Defaults to ``"sdp"``.
    **kwargs
        Hyperparameters of ``method`` (see :func:`get_method_hyperparameters`).

    Returns
    -------
    flips : np.ndarray
        Per-channel flips in ``{+1, -1}``, shape (n_channels,).
    corr : float
        Correlation between the flipped covariance and ``template_cov`` (the
        metric the methods maximize).
    """
    unknown = set(kwargs) - set(get_method_hyperparameters(method))
    if unknown:
        raise ValueError(
            f"Unknown option(s) {', '.join(map(repr, sorted(unknown)))} for sign "
            f"flipping method {method!r}. Valid options are: "
            f"{', '.join(get_method_hyperparameters(method))}."
        )
    if method == "random":
        # No per-session progress bars in pipeline logs
        kwargs.setdefault("use_tqdm", False)

    flips, metrics = osl_signflip.find_flips(
        cov, template_cov, n_embeddings, method=method, **kwargs
    )

    # For every method, the last metric is that of the returned flips
    return flips, float(metrics[-1])


def sign_flip(
    parcel_data: np.ndarray,
    template_cov: Union[str, np.ndarray],
    n_embeddings: int = 15,
    standardize: bool = True,
    method: str = DEFAULT_METHOD,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sign flip parcel time courses to match a template covariance.

    Finds the per-parcel ``+1``/``-1`` signs whose (time-delay embedded,
    standardized) covariance best matches ``template_cov`` and applies them.
    This is the array-level core shared by :func:`sign_flip_mne_raw` and by
    scripts that already hold the parcel data in memory.

    Parameters
    ----------
    parcel_data : np.ndarray
        Parcel time courses, shape ``(n_parcels, n_samples)``.
    template_cov : str or np.ndarray
        Template covariance to align to, or the path to a ``.npy`` file
        containing it. This must be the covariance of the time-delay embedded
        data (see :func:`compute_covariance`), built with the same
        ``n_embeddings`` and ``standardize`` settings passed here.
    n_embeddings : int, optional
        Number of time-delay embeddings used to build the covariances. Must be odd.
    standardize : bool, optional
        Standardize the embedded data before computing the covariance.
    method : str, optional
        Sign flipping method, one of :data:`METHODS`. Defaults to ``"sdp"``.
    **kwargs
        Hyperparameters of ``method`` (see :func:`get_method_hyperparameters`).

    Returns
    -------
    flipped_data : np.ndarray
        ``parcel_data`` with the per-parcel signs applied, same shape as the
        input.
    flips : np.ndarray
        The per-parcel ``+1``/``-1`` signs that were applied, shape
        ``(n_parcels,)``.
    corr : float
        Correlation between the flipped covariance and ``template_cov`` (the
        metric maximized by the flip search).
    """
    if not isinstance(template_cov, np.ndarray):
        template_cov = np.load(str(template_cov))
    cov = compute_covariance(parcel_data.T, n_embeddings, standardize)
    flips, corr = find_flips(cov, template_cov, n_embeddings, method, **kwargs)
    logger.info(
        f"Flipped {int((flips == -1).sum())}/{len(flips)} parcels "
        f"(template corr {corr:.3f}, {method})"
    )
    return parcel_data * flips[:, np.newaxis], flips, corr


def sign_flip_mne_raw(
    fif,
    template_cov: Union[str, np.ndarray],
    output_file: Optional[str] = None,
    n_embeddings: int = 15,
    standardize: bool = True,
    picks: str = "misc",
    extra_chans: Union[str, list, None] = "stim",
    method: str = DEFAULT_METHOD,
    **kwargs,
) -> Optional["mne.io.Raw"]:
    """Sign flip a parcellated fif file to match a template covariance.

    Convenience wrapper that loads parcel time courses from a fif file (or
    :class:`mne.io.Raw`), finds the per-parcel signs that best align its
    (time-delay embedded) covariance to ``template_cov``, and applies them.
    The result is either saved to a new fif file or returned as an
    :class:`mne.io.Raw` object (see ``output_file``).

    Parameters
    ----------
    fif : str or mne.io.Raw
        Path to a parcellated fif file, or a loaded :class:`mne.io.Raw` object.
        The parcel time courses are read from the ``picks`` channels.
    template_cov : str or np.ndarray
        Template covariance to align to, or the path to a ``.npy`` file
        containing it. This must be the covariance of the time-delay embedded
        data (see :func:`compute_covariance`), built with the same
        ``n_embeddings`` and ``standardize`` settings passed here.
    output_file : str, optional
        Path to save the sign-flipped fif file to. If ``None`` (the default),
        nothing is written to disk and the sign-flipped :class:`mne.io.Raw`
        object is returned instead (see Returns).
    n_embeddings : int, optional
        Number of time-delay embeddings used to build the covariances. Must be odd.
    standardize : bool, optional
        Standardize the embedded data before computing the covariance.
    picks : str, optional
        Channel type holding the parcel time courses. Parcellated fif files
        written by osl-ephys store parcels as ``"misc"`` channels.
    extra_chans : str or list of str, optional
        Extra channels (e.g. ``"stim"``) to copy from the input to the output.
        Passed to :func:`osl_ephys.parcellation.save_as_fif`.
    method : str, optional
        Sign flipping method, one of :data:`METHODS`. Defaults to ``"sdp"``.
    **kwargs
        Hyperparameters of ``method`` (see :func:`get_method_hyperparameters`).

    Returns
    -------
    raw : mne.io.Raw or None
        The sign-flipped data if ``output_file`` is ``None``, otherwise ``None``.

    Examples
    --------
    Save the sign-flipped data to a new fif file::

        from osl_ephys.group import sign_flipping

        sign_flipping.sign_flip_mne_raw(
            "lcmv-parc-raw.fif",
            "template_cov.npy",
            "sflip-lcmv-parc-raw.fif",
        )

    Or get the sign-flipped data back as an :class:`mne.io.Raw` without
    saving, using another method::

        raw = sign_flipping.sign_flip_mne_raw(
            "lcmv-parc-raw.fif", "template_cov.npy", method="spectral"
        )
    """
    if isinstance(fif, mne.io.BaseRaw):
        raw = fif
    else:
        raw = mne.io.read_raw_fif(str(fif), preload=True)
    parcel_data = raw.get_data(picks=picks, reject_by_annotation="omit")
    flipped_data, _, _ = sign_flip(
        parcel_data, template_cov, n_embeddings, standardize, method, **kwargs
    )
    if output_file is None:
        return convert_to_mne_raw(
            flipped_data,
            raw,
            ch_names=[f"parcel_{i}" for i in range(flipped_data.shape[0])],
            extra_chans="stim" if extra_chans is None else extra_chans,
        )
    save_as_fif(flipped_data, raw, filename=output_file, extra_chans=extra_chans)
    return None
