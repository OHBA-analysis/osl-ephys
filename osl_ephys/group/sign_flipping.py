"""Sign flipping.

Source reconstruction leaves the sign of each channel (parcel) arbitrary, so the
same parcel can have opposite polarity across sessions. Before pooling sessions
we align these signs: for each session we search for the ``+1``/``-1`` per-channel
vector whose (time-delay embedded, standardized) covariance best matches a
template.

For the common case of sign flipping a single parcellated fif file (or
:class:`mne.io.Raw`) to match a saved template covariance, see
:func:`sign_flip_mne_raw`.
"""

import logging
from typing import Optional, Tuple, Union

import mne
import numpy as np

from osl_ephys.utils import processing
from osl_ephys.parcellation.parcellation import convert_to_mne_raw, save_as_fif

logger = logging.getLogger("osl_ephys")


def calc_cov(
    array: np.ndarray, n_embeddings: int, standardize: bool = True
) -> np.ndarray:
    """Covariance of time-delay embedded (optionally standardized) data.

    Parameters
    ----------
    array : np.ndarray
        Data, shape (n_samples, n_channels).
    n_embeddings : int
        Number of time-delay embeddings.
    standardize : bool, optional
        Standardize the embedded data before computing the covariance.

    Returns
    -------
    cov : np.ndarray
        Covariance, shape (n_channels * n_embeddings, n_channels * n_embeddings).
    """
    array = processing.time_embed(array, n_embeddings)
    if standardize:
        array = processing.standardize(array, create_copy=False)
    return np.cov(array.T)


def calc_corr(
    M1: np.ndarray, M2: np.ndarray, n_embeddings: int, mode: Optional[str] = None
) -> float:
    """Correlation between the upper triangles of two covariances.

    The first ``n_embeddings`` diagonals are skipped (``k=n_embeddings`` in
    :func:`numpy.triu_indices`) so within-channel, near-lag terms — which do not
    depend on the sign — are excluded from the comparison.

    Parameters
    ----------
    M1, M2 : np.ndarray
        Covariance matrices of the same shape.
    n_embeddings : int
        Number of time-delay embeddings (sets the skipped diagonal offset).
    mode : str, optional
        If ``"abs"``, compare the absolute values (sign-invariant), used when
        picking a median template session.

    Returns
    -------
    corr : float
        Pearson correlation between the selected entries.
    """
    if mode == "abs":
        M1 = np.abs(M1)
        M2 = np.abs(M2)
    m, n = np.triu_indices(M1.shape[0], k=n_embeddings)
    return np.corrcoef([M1[m, n], M2[m, n]])[0, 1]


def apply_flips_to_cov(
    cov: np.ndarray, flips: np.ndarray, n_embeddings: int
) -> np.ndarray:
    """Apply a ``±1`` per-channel flip vector to an embedded covariance."""
    f = np.repeat(flips, n_embeddings)[np.newaxis, ...]
    return cov * (f.T @ f)


class _FlipObjective:
    """Reduced ``O(n_channels²)`` evaluator of the sign-flip correlation.

    The correlation :func:`calc_corr` compares depends on the ``±1`` per-channel
    flip vector ``f`` only through two quadratic forms,

        P(f) = sum_{p<q} f_p f_q A[p, q]      (agreement with the template)
        Q(f) = sum_{p<q} f_p f_q B[p, q]      (mean of the flipped covariance)

    where ``A`` and ``B`` are ``n_channels × n_channels`` matrices built once from
    the selected entries of the off-diagonal covariance blocks. The correlation is
    then a closed-form function of ``P(f)``, ``Q(f)`` and a few constants, so it
    can be evaluated exactly without rebuilding the full covariance. This
    reproduces ``calc_corr(apply_flips_to_cov(cov, f, E), template_cov, E)`` to
    machine precision.

    Parameters
    ----------
    cov, template_cov : np.ndarray
        Session and template embedded covariances (from :func:`calc_cov`).
    n_channels : int
        Number of channels (parcels).
    n_embeddings : int
        Number of time-delay embeddings used to build the covariances.
    """

    def __init__(
        self,
        cov: np.ndarray,
        template_cov: np.ndarray,
        n_channels: int,
        n_embeddings: int,
    ):
        C = np.asarray(cov)
        T = np.asarray(template_cov)
        E = n_embeddings
        N = n_channels * E

        # Validate shapes so misuse fails with a clear message rather than a
        # confusing error deeper in the search.
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError(f"cov must be a square 2D array, got shape {C.shape}.")
        if C.shape != T.shape:
            raise ValueError(
                f"cov {C.shape} and template_cov {T.shape} must have the same shape."
            )
        if C.shape[0] != N:
            raise ValueError(
                f"cov side length {C.shape[0]} must equal "
                f"n_channels * n_embeddings = {n_channels} * {E} = {N}."
            )

        # The exact entries calc_corr compares (strict upper triangle, k=E). All
        # of these fall in off-diagonal channel blocks; within-channel blocks are
        # skipped and do not depend on the sign.
        rows, cols = np.triu_indices(N, k=E)
        c = C[rows, cols]
        t = T[rows, cols]
        pc = rows // E  # channel of each selected row
        qc = cols // E  # channel of each selected column

        A = np.zeros((n_channels, n_channels))
        B = np.zeros((n_channels, n_channels))
        np.add.at(A, (pc, qc), c * t)
        np.add.at(B, (pc, qc), c)
        # Symmetrize; the diagonal stays zero (within-channel entries excluded).
        self.A = A + A.T
        self.B = B + B.T

        self.n_channels = n_channels
        self.S = float(len(rows))  # number of compared entries, |S|
        self.mT = t.sum() / self.S  # mean of the template over S
        self.sigT = np.sqrt((t * t).sum() / self.S - self.mT**2)  # std of template
        self.K = (c * c).sum() / self.S  # mean of C^2 over S (flip-invariant)

        if self.sigT == 0.0 or self.K == 0.0:
            raise ValueError(
                "Degenerate covariance: zero variance across the compared "
                "off-diagonal entries of cov or template_cov."
            )

    def corr_from_PQ(self, P, Q):
        """Correlation given the two quadratic forms ``P``, ``Q``.

        ``P`` and ``Q`` may each be a scalar or an array (the coordinate-ascent
        sweep evaluates every single-channel flip at once).
        """
        mean_CF = Q / self.S
        # Clip tiny negatives from rounding; the true variance is non-negative.
        std_CF = np.sqrt(np.maximum(self.K - mean_CF**2, 0.0))
        num = P / self.S - mean_CF * self.mT
        denom = std_CF * self.sigT
        # denom is 0 only for a (near-)constant flipped covariance; report 0 there
        # so the search sees no correlation instead of propagating nan/inf.
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = num / denom
        return np.where(denom > 0, corr, 0.0)

    def corr(self, f: np.ndarray) -> np.ndarray:
        """Correlation with the template for flip vector ``f``."""
        f = np.asarray(f, dtype=float)
        return self.corr_from_PQ(0.5 * f @ (self.A @ f), 0.5 * f @ (self.B @ f))


def _coordinate_ascent(obj: _FlipObjective, f: np.ndarray) -> Tuple[np.ndarray, float]:
    """Greedy single-flip coordinate ascent on the exact correlation.

    Repeatedly flip the single channel that most increases the correlation until
    no single flip helps. Fields ``u = A f`` and ``v = B f`` are maintained so
    each sweep costs ``O(n_channels)``.
    """
    f = np.asarray(f, dtype=float).copy()
    u = obj.A @ f
    v = obj.B @ f
    P = 0.5 * f @ u
    Q = 0.5 * f @ v
    cur = obj.corr_from_PQ(P, Q)
    while True:
        # Flipping channel p: P -> P - 2 f_p u_p, Q -> Q - 2 f_p v_p.
        P_new = P - 2.0 * f * u
        Q_new = Q - 2.0 * f * v
        cand = obj.corr_from_PQ(P_new, Q_new)
        p = int(np.argmax(cand))
        if cand[p] <= cur + 1e-12:
            break
        df = -2.0 * f[p]
        u += obj.A[:, p] * df
        v += obj.B[:, p] * df
        f[p] = -f[p]
        P, Q, cur = P_new[p], Q_new[p], cand[p]
    return f, float(cur)


def find_flips(
    cov: np.ndarray,
    template_cov: np.ndarray,
    n_channels: int,
    n_embeddings: int,
    n_random_starts: int = 3,
    n_spectral: int = 2,
) -> Tuple[np.ndarray, float]:
    """Find the ``±1`` per-channel flip vector aligning ``cov`` to a template.

    The objective is reduced to a small ``n_channels × n_channels`` problem that
    is exactly the correlation :func:`calc_corr` maximizes. In that form aligning
    the signs is a Z2 (Ising / MAX-CUT) synchronization problem, solved by:

    1. warm starts: the sign of the leading eigenvector(s) of the reduced matrix
       (the spectral relaxation), plus the all-``+1`` start and
       ``n_random_starts`` random restarts;
    2. exact greedy coordinate ascent on the true correlation from each start,
       keeping the best local optimum.

    Parameters
    ----------
    cov : np.ndarray
        This session's embedded covariance (from :func:`calc_cov`).
    template_cov : np.ndarray
        Template embedded covariance to align to (same shape as ``cov``).
    n_channels : int
        Number of channels (parcels).
    n_embeddings : int
        Number of time-delay embeddings used to build ``cov``.
    n_random_starts : int, optional
        Number of random restarts (in addition to the spectral and all-``+1``
        starts). The restarts use a fixed internal seed, so the whole search is
        deterministic and reproducible.
    n_spectral : int, optional
        Number of leading eigenvectors of the reduced matrix used as warm starts.

    Returns
    -------
    flips : np.ndarray
        Per-channel flips in ``{+1, -1}``, shape (n_channels,), dtype int.
    metric : float
        The achieved correlation with ``template_cov``.
    """
    # Fixed seed: the restarts only need to be diverse, not run-to-run random,
    # so the search stays fully deterministic and reproducible.
    rng = np.random.default_rng(0)
    obj = _FlipObjective(cov, template_cov, n_channels, n_embeddings)

    # Candidate starting points: all-+1, spectral warm starts, random restarts.
    starts = [np.ones(n_channels)]
    n_spectral = min(n_spectral, n_channels)
    if n_spectral > 0:
        _, evecs = np.linalg.eigh(obj.A)
        for k in range(1, n_spectral + 1):
            s = np.sign(evecs[:, -k])
            s[s == 0] = 1.0
            starts.append(s)
    for _ in range(n_random_starts):
        starts.append(rng.choice([-1.0, 1.0], size=n_channels).astype(float))

    # Exact coordinate ascent from each start; keep the best local optimum.
    best_flips = np.ones(n_channels)
    best_metric = -np.inf
    for s in starts:
        flips, metric = _coordinate_ascent(obj, s)
        if metric > best_metric:
            best_flips, best_metric = flips, metric

    return best_flips.astype(int), float(best_metric)


def sign_flip(
    parcel_data: np.ndarray,
    template_cov: Union[str, np.ndarray],
    n_embeddings: int = 15,
    standardize: bool = True,
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
        data (see :func:`calc_cov`), built with the same ``n_embeddings`` and
        ``standardize`` settings passed here.
    n_embeddings : int, optional
        Number of time-delay embeddings used to build the covariances.
    standardize : bool, optional
        Standardize the embedded data before computing the covariance.

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
        metric maximised by the flip search).
    """
    print()
    print("Sign flipping")
    print("-------------")

    if not isinstance(template_cov, np.ndarray):
        template_cov = np.load(str(template_cov))

    cov = calc_cov(parcel_data.T, n_embeddings, standardize)
    flips, corr = find_flips(
        cov,
        template_cov,
        n_channels=parcel_data.shape[0],
        n_embeddings=n_embeddings,
    )
    print(
        f"Flipped {(flips == -1).sum()}/{len(flips)} parcels "
        f"(template corr {corr:.3f})"
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
) -> Union[Tuple[np.ndarray, float], "mne.io.Raw"]:
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
        data (see :func:`calc_cov`), built with the same ``n_embeddings`` and
        ``standardize`` settings passed here.
    output_file : str, optional
        Path to save the sign-flipped fif file to. If ``None`` (the default),
        nothing is written to disk and the sign-flipped :class:`mne.io.Raw`
        object is returned instead (see Returns).
    n_embeddings : int, optional
        Number of time-delay embeddings used to build the covariances.
    standardize : bool, optional
        Standardize the embedded data before computing the covariance.
    picks : str, optional
        Channel type holding the parcel time courses. Parcellated fif files
        written by osl-ephys store parcels as ``"misc"`` channels.
    extra_chans : str or list of str, optional
        Extra channels (e.g. ``"stim"``) to copy from the input to the output.
        Passed to :func:`osl_ephys.parcellation.save_as_fif`.

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
    saving::

        raw = sign_flipping.sign_flip_mne_raw("lcmv-parc-raw.fif", "template_cov.npy")
    """
    if isinstance(fif, mne.io.BaseRaw):
        raw = fif
    else:
        raw = mne.io.read_raw_fif(str(fif), preload=True)

    parcel_data = raw.get_data(picks=picks, reject_by_annotation="omit")

    flipped_data, _, _ = sign_flip(parcel_data, template_cov, n_embeddings, standardize)

    if output_file is None:
        return convert_to_mne_raw(
            flipped_data,
            raw,
            ch_names=[f"parcel_{i}" for i in range(flipped_data.shape[0])],
            extra_chans="stim" if extra_chans is None else extra_chans,
        )

    save_as_fif(flipped_data, raw, filename=output_file, extra_chans=extra_chans)


def _apply_flip_convention(flips):
    """Fixes the arbitrary global sign by making the sum of flips positive.

    The global sign is a gauge freedom where ``flips`` and ``-flips`` produce the same
    flipped covariance. Hence, a deterministic convention is adopted to make the result
    reproducible.

    Parameters
    ----------
    flips : numpy.ndarray
        Vector of 1s and -1s (zeros, e.g. from ``np.sign``, are treated as +1).
        Shape is (n_channels,).

    Returns
    -------
    flips : numpy.ndarray
        Vector of 1s and -1s with a non-negative sum.
        Shape is (n_channels,).
    """
    flips = np.asarray(flips, dtype=float).copy()
    flips[flips == 0] = 1.0  # np.sign() can emit 0; a 0 "flip" would zero the channel
    if np.sum(flips) < 0:
        flips *= -1.0
    return flips


def _solve_sdp_mixing(W, rank=None, n_restarts=5, max_iter=1000, tol=1e-8, rng=None):
    """Solves the Z2 synchronization problem using SDP with the low-rank mixing method.

    The semidefinite relaxation of the sign-agreement problem is

        maximize  <W, X>   subject to   X >= 0 (positive semidefinite),  diag(X) = 1,

    where ``X`` relaxes the rank-one product ``flips @ flips.T``. We solve it with the
    mixing method (Wang, Chang & Kolter, 2017): a Burer-Monteiro factorization
    ``X = V @ V.T`` with unit-norm rows ``v_i`` in R^rank, updated by the cyclic rule
    ``v_i <- normalize(sum_{j != i} W[i, j] v_j)``. Each update is coordinate-wise
    optimal, so ``<W, X>`` increases monotonically; at ``rank >= sqrt(2n)`` all local
    optima are global, so the returned value is (a tight estimate of) the SDP optimum.

    Parameters
    ----------
    W : numpy.ndarray
        Symmetric (n_channels, n_channels) affinity matrix with a zero diagonal.
    rank : int
        Factorization rank. Defaults to ``ceil(sqrt(2n)) + 1``, which guarantees the
        global-optimality property above.
    n_restarts : int
        Number of random restarts wherein the best objective is kept.
    max_iter : int
        Maximum number of coordinate-descent sweeps per restart.
    tol : float
        Convergence tolerance on the largest row update within a sweep.
    rng : numpy.random.Generator
        Random generator used for the restarts (for reproducibility).

    Returns
    -------
    value : float
        Best SDP objective ``<W, X>`` over the restarts.
    V : numpy.ndarray
        (n_channels, rank) factor of the best solution; rows are unit-norm.
    """
    # Validate inputs
    n = W.shape[0]
    if rank is None:
        rank = int(np.ceil(np.sqrt(2 * n))) + 1
    if rng is None:
        rng = np.random.default_rng(0)
    eps = 1e-12

    # Apply SDP solver
    best_value = -np.inf
    best_V = None
    for _ in range(int(n_restarts)):
        V = rng.standard_normal((n, rank))
        V /= np.linalg.norm(V, axis=1, keepdims=True) + eps
        for _ in range(int(max_iter)):
            max_change = 0.0
            for i in range(n):
                g = (
                    W[i] @ V
                )  # sum_j W[i, j] v_j (the diagonal term vanishes as W[i, i] = 0)
                g_norm = np.linalg.norm(g)
                if g_norm > eps:
                    new_v = g / g_norm
                    max_change = max(max_change, np.linalg.norm(new_v - V[i]))
                    V[i] = new_v
            if max_change < tol:
                break
        value = float(np.sum(W * (V @ V.T)))
        if value > best_value:
            best_value = value
            best_V = V

    return best_value, best_V


def _gw_round(W, V, n_samples=500, rng=None):
    """Rounds an SDP factor to a flip vector using Goemans-Williamson hyperplane rounding.

    Given the factor ``V`` of the SDP solution, we sample random hyperplanes ``r`` and read
    off the cut ``f_i = sign(v_i . r)``, keeping the sample that maximises ``f.T @ W @ f``.
    This recovers a discrete +/-1 assignment from the continuous relaxation.

    Parameters
    ----------
    W : numpy.ndarray
        Symmetric (n_channels, n_channels) affinity matrix.
    V : numpy.ndarray
        (n_channels, rank) SDP factor from :func:`_solve_sdp_mixing`.
    n_samples : int
        Number of random hyperplanes to try.
    rng : numpy.random.Generator
        Random generator used for the hyperplanes (for reproducibility).

    Returns
    -------
    flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s (sign convention applied).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    best_flips = None
    best_value = -np.inf
    for _ in range(int(n_samples)):
        r = rng.standard_normal(V.shape[1])
        f = np.sign(V @ r)
        f[f == 0] = 1.0
        value = float(f @ W @ f)
        if value > best_value:
            best_value = value
            best_flips = f

    return _apply_flip_convention(best_flips)


class _FlipMetricEvaluator:
    """Evaluator of the sign-flip correlation metric.

    The covariance correlation metric depends on the ``+/-1`` flip vector ``f`` only through
    two quadratic forms, ``P(f) = 0.5 * f @ M @ f`` (agreement with the template) and
    ``Q(f) = 0.5 * f @ S @ f`` (sum of the flipped entries), where ``M`` is the metric-exact
    magnitude affinity and ``S`` the metric-exact subject-only affinity. The exact metric
    is then a closed-form function of ``P``, ``Q`` and a few template constants, so it can be
    evaluated without rebuilding the full covariance. This reproduces
    ``covariance_matrix_correlation(apply_flips_to_covariance(cov, f, E), template_cov, E)``
    to machine precision, which is what lets :func:`_greedy_refine` score every single-channel
    flip in ``O(n_channels)`` by maintaining the fields ``M @ f`` and ``S @ f`` incrementally.

    Magnitude affinity: ``M_ij = sum_block(cov * template)``.

        This is the channel-by-channel coupling fed to the Z2 (sign) solvers: ``M[i, j]``
        measures how much channels ``i`` and ``j`` agree between subject and template, and
        the quadratic form ``flips.T @ M @ flips`` is, up to a flip-invariant normalisation,
        exactly the numerator of the covariance correlation metric. Keeping the raw block inner
        product (rather than a cosine-normalised one) preserves the magnitude information that
        weights strongly-coupled pairs above weak, noisy ones.

    Subject affinity: ``S_ij = sum_block(cov)``.

    NOTE: Both are symmetric (n_channels, n_channels) matrices with a zero diagonal.

    Parameters
    ----------
    cov : numpy.ndarray
        Subject covariance. Shape is (n_channels * n_embeddings, n_channels * n_embeddings).
    template_cov : numpy.ndarray
        Template covariance. Shape is (n_channels * n_embeddings, n_channels * n_embeddings).
    n_embeddings : int
        Number of time-delay embeddings used to build the covariances.
    """

    def __init__(self, cov, template_cov, n_embeddings):
        # Ensure inputs are numpy arrays
        cov = np.asarray(cov)
        template_cov = np.asarray(template_cov)

        # Validate inputs
        if cov.shape != template_cov.shape:
            raise ValueError("cov and template_cov must have the same shape.")
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("cov and template_cov must both be square matrices.")
        if cov.shape[0] % n_embeddings != 0:
            raise ValueError("cov shape is not compatible with n_embeddings.")

        # Get data dimensions
        D = cov.shape[0]
        E = n_embeddings
        n_channels = D // E

        # Get the entries the metric compares
        rows, cols = np.triu_indices(D, k=E)  # strict upper triangle with offset E
        cov_vals = cov[rows, cols]
        tmpl_vals = template_cov[rows, cols]
        row_chan = rows // E  # channel each scored row belongs to
        col_chan = cols // E  # channel each scored column belongs to

        # Compute affinity matrices
        def _accumulate(values):
            out = np.zeros((n_channels, n_channels), dtype=float)
            np.add.at(out, (row_chan, col_chan), values)
            out += out.T  # the diagonal stays zero (row_chan < col_chan throughout)
            return out

        self.M = _accumulate(cov_vals * tmpl_vals)  # magnitude affinity
        self.S = _accumulate(cov_vals)  # subject affinity

        # Get scalar statistics of the scored entries
        self.N = float(len(rows))  # number of scored entries
        self.tmpl_mean = np.mean(tmpl_vals)  # mean template entries
        self.tmpl_std = np.std(tmpl_vals)  # standard deviation of template entries
        self.cov_mean_sq = float(
            (cov_vals**2).sum() / self.N
        )  # mean squared subject entries

    def correlation_from_PQ(self, P, Q):
        """Computes Pearson correlation between the flipped subject covariance and the
        template covariance (over the scored entries).

        This function gives exact correlation given the two quadratic forms ``P`` and ``Q``.
        ``P`` and ``Q`` may each be a scalar or an array (the best-improvement sweep scores
        every single-channel flip at once). A (near-)constant flipped covariance gives a zero
        denominator; we report 0 there rather than propagate ``nan``.
        """
        mean_CF = Q / self.N
        std_CF = np.sqrt(np.maximum(self.cov_mean_sq - mean_CF**2, 0.0))
        num = P / self.N - mean_CF * self.tmpl_mean
        denom = std_CF * self.tmpl_std
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = num / denom
        return np.where(denom > 0, corr, 0.0)


def _greedy_refine(
    cov,
    template_cov,
    flips,
    n_embeddings,
    n_refine,
    strategy="best",
    tol=1e-12,
):
    """Greedy coordinate-ascent refinement of a flip vector.

    Starting from ``flips``, we flip the sign of channels whenever doing so increases the
    covariance correlation metric. Unlike the affinity-based solvers, this optimises the
    exact evaluation metric, so it recovers the (centering and normalisation) terms those
    solvers approximate away. Each sweep costs ``O(n_channels)`` rather than
    ``O(n_channels * D**2)``, as it never touches the full covariance while scoring the
    identical exact metric.

    Two strategies are available:

    - ``"first"`` (first-improvement / Gauss-Seidel): sweep over the channels and apply each
      improving single-channel flip as soon as it is found. Cheaper, but the result depends on
      the (fixed) channel order, since later channels are evaluated against the already-updated
      flips.
    - ``"best"`` (best-improvement / steepest ascent): evaluate every single-channel flip from
      the current state and apply only the single best one, repeating until no flip improves.
      Order-invariant (the chosen flip does not depend on channel order) and usually slightly
      better, at the cost of more metric evaluations.

    Both converge to a 1-flip local optimum: a state where no single-channel flip improves the
    metric.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix we would like to sign flip.
    template_cov : numpy.ndarray
        Template covariance matrix.
    flips : numpy.ndarray
        Initial (n_channels,) array of 1s and -1s to refine.
    n_embeddings : int
        Number of time-delay embeddings.
    n_refine : int
        For ``strategy="first"``, the maximum number of sweeps over the channels (the loop
        stops early once a full sweep makes no improvement). For ``strategy="best"``, ``n_refine``
        only scales a generous safety cap that is not reached in practice.
    strategy : str
        Either ``"first"`` or ``"best"`` (default).
    tol : float
        A flip is accepted only if it improves the metric by more than ``tol``. This guards
        against float-noise cycling.

    Returns
    -------
    flips : numpy.ndarray
        Refined (n_channels,) array of 1s and -1s.
    metric : float
        Covariance correlation metric of the refined flips.
    """
    if strategy not in ("first", "best"):
        raise ValueError(f"strategy must be 'first' or 'best', got '{strategy}'.")

    obj = _FlipMetricEvaluator(cov, template_cov, n_embeddings)
    M, S = obj.M, obj.S

    f = _apply_flip_convention(flips).astype(float)
    n_channels = len(f)
    u = M @ f
    v = S @ f
    P = 0.5 * (f @ u)
    Q = 0.5 * (f @ v)
    cur = float(obj.correlation_from_PQ(P, Q))

    def _accept(k, P_k, Q_k, cand):
        # Flip channel k and update the running fields / quadratic forms in O(n_channels)
        nonlocal P, Q, cur
        df = -2.0 * f[k]
        u[:] += M[:, k] * df
        v[:] += S[:, k] * df
        f[k] = -f[k]
        P, Q, cur = float(P_k), float(Q_k), float(cand)

    if strategy == "first":
        # First-improvement: apply each improving single-channel flip as it is found
        for _ in range(int(n_refine)):
            improved = False
            for k in range(n_channels):
                P_k = P - 2.0 * f[k] * u[k]
                Q_k = Q - 2.0 * f[k] * v[k]
                cand = float(obj.correlation_from_PQ(P_k, Q_k))
                if cand > cur + tol:
                    _accept(k, P_k, Q_k, cand)
                    improved = True
            if not improved:
                break
    else:
        # Best-improvement: evaluate all single-channel flips, apply only the best one
        # The cap is a safety bound; the loop converges (and breaks) well before it.
        for _ in range(int(n_refine) * n_channels):
            P_new = P - 2.0 * f * u
            Q_new = Q - 2.0 * f * v
            cand = obj.correlation_from_PQ(P_new, Q_new)
            k_best = int(np.argmax(cand))
            if cand[k_best] > cur + tol:
                _accept(k_best, P_new[k_best], Q_new[k_best], cand[k_best])
            else:
                break

    return _apply_flip_convention(f), cur


def find_flips_spectral(
    cov,
    template_cov,
    n_embeddings,
    refine=True,
    n_refine=5,
    refine_strategy="best",
):
    """Finds channels to flip using spectral synchronization.

    This is a deterministic solver for the underlying Z2 (sign) synchronization
    problem: given the channel-by-channel affinity matrix, the signs of the leading
    eigenvector maximize the quadratic agreement ``flips.T @ W @ flips`` over the unit
    sphere, which is the standard convex relaxation of the discrete sign assignment.
    It is fast and reproducible, with no random initializations. By default the spectral
    solution is then polished with greedy refinement, which optimizes the exact metric.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix we would like to sign flip.
    template_cov : numpy.ndarray
        Template covariance matrix.
    n_embeddings : int
        Number of time-delay embeddings.
    refine : bool
        Should we polish the spectral solution with greedy refinement?
    n_refine : int
        Maximum number of greedy refinement sweeps (only used if ``refine=True``).
    refine_strategy : str
        Greedy refinement strategy, ``"first"`` or ``"best"`` (default).
        See :func:`_greedy_refine`.

    Returns
    -------
    best_flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s indicating whether or not to flip a channel.
    metrics : numpy.ndarray
        Evaluation metric (correlation between covariance matrices) before and after
        flipping. Shape is (2,).
    """
    logger.info("find_flips_spectral")

    # Get the affinity matrix
    W = _FlipMetricEvaluator(cov, template_cov, n_embeddings).M

    # Signs of the eigenvector with the largest eigenvalue give the spectral solution
    eigvals, eigvecs = np.linalg.eigh(W)
    v = eigvecs[:, np.argmax(eigvals)]
    flips = _apply_flip_convention(np.sign(v))

    # Apply flips and evaluate the metric
    metric_before = covariance_matrix_correlation(cov, template_cov, n_embeddings)
    if refine:
        flips, metric_after = _greedy_refine(
            cov, template_cov, flips, n_embeddings, n_refine, strategy=refine_strategy
        )
    else:
        new_cov = apply_flips_to_covariance(cov, flips, n_embeddings)
        metric_after = covariance_matrix_correlation(
            new_cov, template_cov, n_embeddings
        )
    metrics = np.array([metric_before, metric_after], dtype=float)

    logger.info(f"metric before: {metric_before}, after: {metric_after}")

    return flips, metrics


def find_flips_belief_propagation(
    cov,
    template_cov,
    n_embeddings,
    n_iter=200,
    beta=2.0,
    damping=0.5,
    prior_strength=1e-3,
    tol=1e-6,
    refine=True,
    n_refine=5,
    refine_strategy="best",
):
    """Finds channels to flip using loopy belief propagation.

    This is a deterministic solver for the Z2 (sign) synchronization problem by applying
    message passing on an Ising-style model whose pairwise couplings are the channel
    affinity matrix. Belief propagation on a fully-connected graph is approximate (it may
    not converge and can find a local optimum), but it provides a deterministic, reproducible
    alternative to the random search. By default the solution is then polished with greedy
    refinement, which optimizes the exact metric.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix we would like to sign flip.
    template_cov : numpy.ndarray
        Template covariance matrix.
    n_embeddings : int
        Number of time-delay embeddings.
    n_iter : int
        Maximum number of message-passing iterations.
    beta : float
        Inverse temperature. Larger values give more decisive updates.
    damping : float
        Damping factor in [0, 1). Higher values are slower but more stable.
    prior_strength : float
        Small deterministic symmetry-breaking term. Without it, the zero-field model
        can stay at the trivial zero-message fixed point.
    tol : float
        Convergence tolerance on the message updates.
    refine : bool
        Should we polish the belief propagation solution with greedy refinement?
    n_refine : int
        Maximum number of greedy refinement sweeps (only used if ``refine=True``).
    refine_strategy : str
        Greedy refinement strategy, ``"first"`` or ``"best"`` (default).
        See :func:`_greedy_refine`.

    Returns
    -------
    best_flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s indicating whether or not to flip a channel.
    metrics : numpy.ndarray
        Evaluation metric (correlation between covariance matrices) as a function of
        iterations, including the initial metric and (if ``refine=True``) a final entry
        for the refined solution. The history is padded to a fixed length so every
        subject in a run returns the same shape. Shape is (n_iter + 2,) if ``refine``
        else (n_iter + 1,).
    """
    logger.info("find_flips_belief_propagation")

    # Set numeric types
    n_iter = int(n_iter)
    beta = float(beta)
    damping = float(damping)
    prior_strength = float(prior_strength)
    tol = float(tol)

    # Get the affinity matrix
    obj = _FlipMetricEvaluator(cov, template_cov, n_embeddings)
    W = obj.M
    n = W.shape[0]  # number of channels
    eps = 1e-12

    # Normalize the coupling scale so that tanh(beta * W) is well-behaved
    w_scale = np.max(np.abs(W))
    if w_scale > eps:
        W = W / w_scale
    # NOTE: If not normalized, the magnitude affinity is otherwise unbounded. This keeps the
    #       relative couplings but gives beta a consistent meaning.

    # Deterministic tiny unary bias to break symmetry
    row_strength = np.sum(W, axis=1)
    scale = np.max(np.abs(row_strength))
    if scale < eps:
        h = np.zeros(n, dtype=float)
    else:
        h = prior_strength * row_strength / scale
    # NOTE: This bias keeps the method reproducible and avoids the trivial zero fixed point.

    # Preallocate messages u[i, j] (represents the cavity influence from i -> j)
    u = np.zeros((n, n), dtype=float)  # store full dense matrix for simplicity
    tanh_betaW = np.tanh(beta * W)

    # Score the metric history through the exact evaluator
    def metric_of(f):
        P = 0.5 * f @ (obj.M @ f)
        Q = 0.5 * f @ (obj.S @ f)
        return float(obj.correlation_from_PQ(P, Q))

    # NOTE: `correlation_from_PQ` reproduces `covariance_matrix_correlation` to machine
    # precision but costs only O(n_channels**2) per iteration instead of rebuilding the
    # full covariance.

    metric_before = covariance_matrix_correlation(cov, template_cov, n_embeddings)
    metrics = [metric_before]
    flips = _apply_flip_convention(np.ones(n))

    for _ in range(n_iter):
        # Cavity field for every ordered pair: cavity[i, j] = h[i] + incoming[i] - u[j, i]
        incoming = np.sum(u, axis=0)  # sum_k u[k, i] for each i
        cavity = (h + incoming)[:, np.newaxis] - u.T

        # Ising BP update, vectorised over all pairs
        # (This is the vectorised equivalent of looping over all ordered pairs (i, j) with i != j.)
        tanh_prod = np.clip(tanh_betaW * np.tanh(cavity), -1.0 + eps, 1.0 - eps)
        new_u = np.arctanh(tanh_prod)
        np.fill_diagonal(new_u, 0.0)  # no self-messages (i == j)

        # Damping
        u_next = damping * u + (1.0 - damping) * new_u
        delta = np.max(np.abs(u_next - u))
        u = u_next

        # Current beliefs / node fields
        beliefs = h + np.sum(u, axis=0)
        flips = _apply_flip_convention(np.sign(beliefs))
        metrics.append(metric_of(flips))

        if delta < tol:
            break

    # Optionally polish with greedy refinement
    if refine:
        flips, metric_refined = _greedy_refine(
            cov, template_cov, flips, n_embeddings, n_refine, strategy=refine_strategy
        )
        metrics.append(metric_refined)

    logger.info(f"metric before: {metrics[0]}, after: {metrics[-1]}")

    # Pad the history to a fixed length so all subjects return the same shape, which
    # the report needs to stack the per-subject metrics into a single array
    target_len = n_iter + (2 if refine else 1)
    metrics = metrics + [metrics[-1]] * (target_len - len(metrics))
    metrics = np.asarray(metrics, dtype=float)

    return flips, metrics


def find_flips_sdp(
    cov,
    template_cov,
    n_embeddings,
    sdp_rank=None,
    sdp_n_restarts=5,
    sdp_max_iter=1000,
    sdp_tol=1e-8,
    gw_samples=500,
    refine=True,
    n_refine=5,
    refine_strategy="best",
    random_state=0,
):
    """Finds channels to flip using a semidefinite relaxation.

    This is a deterministic solver for the Z2 (sign) synchronization problem that employs
    the tighter semidefinite relaxation

        maximize  <W, X>   subject to   X >= 0 (positive semidefinite),  diag(X) = 1,

    where ``X`` relaxes the rank-one product ``flips @ flips.T``. Spectral synchronization
    is the looser relaxation that keeps only ``trace(X) = n``, so the SDP is at least as tight.
    The relaxation is solved with the low-rank mixing method, and the continuous solution is
    rounded back to +/-1 by Goemans-Williamson hyperplane rounding. By default the solution is
    then polished with greedy refinement, which optimizes the exact metric.

    NOTE: The mixing method and hyperplane rounding use randomness (restarts and random
    hyperplanes); ``random_state`` seeds them so the result is reproducible.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix we would like to sign flip.
    template_cov : numpy.ndarray
        Template covariance matrix.
    n_embeddings : int
        Number of time-delay embeddings.
    sdp_rank : int
        Rank of the mixing-method factorisation. Defaults to ``ceil(sqrt(2 n)) + 1``.
    sdp_n_restarts : int
        Number of random restarts for the mixing method.
    sdp_max_iter : int
        Maximum number of coordinate-descent sweeps per restart.
    sdp_tol : float
        Convergence tolerance for the mixing method.
    gw_samples : int
        Number of random hyperplanes for Goemans-Williamson rounding.
    refine : bool
        Should we polish the rounded solution with greedy refinement?
    n_refine : int
        Maximum number of greedy refinement sweeps (only used if ``refine=True``).
    refine_strategy : str
        Greedy refinement strategy, ``"first"`` (first-improvement) or
        ``"best"`` (best-improvement / steepest ascent, default).
        See :func:`_greedy_refine`.
    random_state : int
        Seed for the mixing-method restarts and hyperplane rounding, so the result is
        reproducible.

    Returns
    -------
    best_flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s indicating whether or not to flip a channel.
    metrics : numpy.ndarray
        Evaluation metric (correlation between covariance matrices) before and after
        flipping. Shape is (2,).
    """
    logger.info("find_flips_sdp")

    # Set random seed
    rng = np.random.default_rng(random_state)

    # Get the affinity matrix
    W = _FlipMetricEvaluator(cov, template_cov, n_embeddings).M

    # Solve the SDP relaxation and round the continuous solution back to +/-1
    _, V = _solve_sdp_mixing(
        W,
        rank=sdp_rank,
        n_restarts=sdp_n_restarts,
        max_iter=sdp_max_iter,
        tol=sdp_tol,
        rng=rng,
    )
    flips = _gw_round(W, V, n_samples=gw_samples, rng=rng)

    # Apply flips and evaluate the metric
    metric_before = covariance_matrix_correlation(cov, template_cov, n_embeddings)
    if refine:
        flips, metric_after = _greedy_refine(
            cov, template_cov, flips, n_embeddings, n_refine, strategy=refine_strategy
        )
    else:
        new_cov = apply_flips_to_covariance(cov, flips, n_embeddings)
        metric_after = covariance_matrix_correlation(
            new_cov, template_cov, n_embeddings
        )
    metrics = np.array([metric_before, metric_after], dtype=float)

    logger.info(f"metric before: {metric_before}, after: {metric_after}")

    return flips, metrics


def covariance_matrix_correlation(M1, M2, diag_offset=0, mode=None):
    """Calculates the Pearson correlation between covariance matrices.

    Parameters
    ----------
    M1 : numpy.ndarray
        First covariance matrix.
    M2 : numpy.ndarray
        Second covariance matrix.
    diag_offset : int
        To calculate the distance we take the upper triangle.
        This argument allows us to specify an offset from the diagonal
        so we can choose not to take elements near the diagonal.
    mode : str
        Either 'abs', 'sign' or None.
    """
    if mode == "abs":
        M1 = np.abs(M1)
        M2 = np.abs(M2)
    elif mode == "sign":
        M1 = np.sign(M1)
        M2 = np.sign(M2)

    # Get the upper triangles
    i, j = np.triu_indices(M1.shape[0], k=diag_offset)
    M1 = M1[i, j]
    M2 = M2[i, j]

    # Calculate correlation
    return np.corrcoef([M1, M2])[0, 1]


def apply_flips_to_covariance(cov, flips, n_embeddings=1):
    """Applies flips to a covariance matrix.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix to apply flips to.
        Shape must be (n_channels*n_embeddings, n_channels*n_embeddings).
    flips : numpy.ndarray
        Vector of 1s and -1s indicating whether or not to flip a channels.
        Shape must be (n_channels,).
    n_embeddings : int
        Number of embeddings used when calculating the covariance.

    Returns
    -------
    cov : numpy.ndarray
        Flipped covariance matrix.
    """
    # flips is a (n_channels,) array however the covariance matrix is (n_channels*n_embeddings, n_channels*n_embeddings),
    # we repeat the flips vector to account for the extra channels due to time embedding
    flips = np.repeat(flips, n_embeddings)[np.newaxis, ...]
    flips = flips.T @ flips
    return cov * flips
