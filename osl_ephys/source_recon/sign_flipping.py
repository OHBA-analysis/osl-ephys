"""Functions for fixing the dipole sign ambiguity of beamformed data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          SungJun Cho <sungjun.cho@ndcn.ox.ac.uk>

import os.path as op

import mne
import numpy as np
from tqdm import trange

from osl_ephys.utils.logger import log_or_print


def _get_parc_chans(raw):
    """Gets parcel channel names in an mne.Raw or mne.Epochs object.

    Parameters
    ----------
    raw : mne.Raw or mne.Epochs
        Raw or Epochs object.

    Returns
    -------
    parc_chans : list of str or str
        Parcel channel names. If no channels called 'parcel_X' are found
        in the raw object, then we return 'misc'.
    """
    # Parcel channels are those called 'parcel_X'
    parc_chans = [ch for ch in raw.ch_names if "parcel" in ch]
    if len(parc_chans) == 0:
        # Old parc-raw.fif didn't use the 'parcel_X' naming convention for parcel channels,
        # so we select all misc channels for backwards compatibility
        parc_chans = "misc"
    return parc_chans


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
                g = W[i] @ V  # sum_j W[i, j] v_j (the diagonal term vanishes as W[i, i] = 0)
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
        self.cov_mean_sq = float((cov_vals ** 2).sum() / self.N)  # mean squared subject entries

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


def find_flips(
    cov,
    template_cov,
    n_embeddings,
    n_restarts,
    n_iter,
    max_flips,
    use_tqdm=True,
    random_state=0,
):
    """Finds channels to flip.

    We search for the channels to flip by randomly flipping them and saving the
    flips that maximise the correlation of the covariance matrices between subjects.

    Parameters
    ----------
    cov : numpy.ndarray
        Covariance matrix we would like to sign flip.
    template_cov : numpy.ndarray
        Template covariance matrix.
    n_embeddings : int
        Number of time-delay embeddings.
    n_restarts : int
        Number of independent random searches to perform.
    n_iter : int
        Number of sign flipping iterations per subject to perform.
    max_flips : int
        Maximum number of channels to flip in an iteration.
    use_tqdm : bool
        Should we display a tqdm progress bar?
    random_state : int
        Seed for the random search, so the result is reproducible (default 0). Pass ``None``
        for an unseeded generator, in which case the search is not reproducible.

    Returns
    -------
    best_flips : numpy.ndarray
        A (n_channels,) array of 1s and -1s indicating whether or not to flip a channels.
    metrics : numpy.ndarray
        Evaluation metric (correlation between covariance matrices) as a function of
        iterations. Shape is (n_iter + 1,).
    """
    log_or_print("find_flips")

    # Set random seed
    rng = np.random.default_rng(random_state)

    # Get the number of channels
    n_channels = cov.shape[-1] // n_embeddings

    # Validation
    if max_flips > n_channels:
        raise ValueError(f"max_flips ({max_flips}) must be less than the number of channels ({n_channels})")

    # Find the best channels to flip
    best_flips = np.ones(n_channels)
    best_metric = 0
    metrics = []
    for n in range(n_restarts):
        # Reset the flips and calculate the evaluation metric before sign flipping
        flips = np.ones(n_channels)
        metric = covariance_matrix_correlation(cov, template_cov, n_embeddings)
        if n == 0:
            metrics.append(metric)
            log_or_print(f"restart #{n}, unflipped metric: {metric}")

        # Randomly permute the sign of different channels and calculate the metric
        if use_tqdm:
            iterator = trange(n_iter, desc="sign flipping")
        else:
            iterator = range(n_iter)
        for _ in iterator:
            new_flips = randomly_flip(flips, max_flips, rng=rng)
            new_cov = apply_flips_to_covariance(cov, new_flips, n_embeddings)
            new_metric = covariance_matrix_correlation(new_cov, template_cov, n_embeddings)
            if new_metric > metric:
                # We've found an improved solution, let's save it
                flips = new_flips
                metric = new_metric

        # Update best_flips if this was the best search
        if metric > best_metric:
            best_flips = flips
            best_metric = metric

        # Save metric as a function of restarts
        metrics.append(best_metric)
        log_or_print(f"restart #{n}, current best metric: {best_metric}")

    return best_flips, metrics


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
    log_or_print("find_flips_spectral")

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
        metric_after = covariance_matrix_correlation(new_cov, template_cov, n_embeddings)
    metrics = np.array([metric_before, metric_after], dtype=float)

    log_or_print(f"metric before: {metric_before}, after: {metric_after}")

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
    log_or_print("find_flips_belief_propagation")

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

    log_or_print(f"metric before: {metrics[0]}, after: {metrics[-1]}")

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
    log_or_print("find_flips_sdp")

    # Set random seed
    rng = np.random.default_rng(random_state)

    # Get the affinity matrix
    W = _FlipMetricEvaluator(cov, template_cov, n_embeddings).M

    # Solve the SDP relaxation and round the continuous solution back to +/-1
    _, V = _solve_sdp_mixing(
        W, rank=sdp_rank, n_restarts=sdp_n_restarts, max_iter=sdp_max_iter, tol=sdp_tol, rng=rng
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
        metric_after = covariance_matrix_correlation(new_cov, template_cov, n_embeddings)
    metrics = np.array([metric_before, metric_after], dtype=float)

    log_or_print(f"metric before: {metric_before}, after: {metric_after}")

    return flips, metrics


def load_covariances(parc_files, n_embeddings=1, standardize=True, loader=None, use_tqdm=True):
    """Loads data and returns its covariance matrix.

    Parameters
    ----------
    parc_files : list of str
        List of paths to parcellated data files to load.
    n_embeddings : int
        Number of time-delay embeddings to perform.
    standardize : bool
        Should we standardize the data?
    loader : function
        Custom function to load parcellated data files.
    use_tqdm : bool
        Should we display a tqdm progress bar?

    Returns
    -------
    covs : numpy.ndarray
        Covariance matrices.
    """
    covs = []
    if use_tqdm:
        iterator = trange(len(parc_files), desc="Calculating covariances")
    else:
        iterator = range(len(parc_files))
    for i in iterator:
        # Load data
        if loader is not None:
            # Use the loader that has been passed
            x = loader(parc_files[i])
        elif "raw.fif" in parc_files[i]:
            # We assume this is a parc-raw.fif file created in beamform_and_parcellated
            raw = mne.io.read_raw_fif(parc_files[i], verbose=False)
            x = raw.get_data(picks=_get_parc_chans(raw), reject_by_annotation="omit", verbose=False)
            x = x.T  # (channels, time) -> (time, channels)
        elif "epo.fif" in parc_files[i]:
            # We assume this is a parc-epo.fif file created in beamform_and_parcellated
            epochs = mne.read_epochs(parc_files[i], verbose=False)
            x = epochs.get_data(picks=_get_parc_chans(epochs))  # (epochs, channels, time)
            x = np.swapaxes(x, 1, 2)
            x = x.reshape(-1, x.shape[-1])  # (time, channels)
        else:
            raise ValueError("Don't know how to load the parcellated data. Please pass loader.")

        # Prepare
        x = time_embed(x, n_embeddings)
        if standardize:
            x = std_data(x)

        # Calculate the covariance
        covs.append(np.cov(x, rowvar=False))

    return np.array(covs)


def find_template_subject(covs, diag_offset=0):
    """Finds a good template subject to use to align dipoles.

    We select the median subject after calculating the similarity between the
    covariances of each subject.

    Parameters
    ----------
    covs : numpy.ndarray
        Covariance of each subject. Shape much be (n_subjects, n_channels, n_channels).
    diag_offset : int
        Offset to apply when getting the upper triangle of the covariance matrix before
        calculating the correlation between covariances.

    Returns
    -------
    index : int
        Index for the template subject.
    """
    # Calculate the similarity between subjects
    n_subjects = len(covs)
    metric = np.zeros([n_subjects, n_subjects])
    for i in trange(n_subjects, desc="Comparing subjects"):
        for j in range(i + 1, n_subjects):
            metric[i, j] = covariance_matrix_correlation(covs[i], covs[j], diag_offset, mode="abs")
            metric[j, i] = metric[i, j]

    # Get the median subject
    metric_sum = np.sum(metric, axis=1)
    argmedian = np.argsort(metric_sum)[len(metric_sum) // 2]

    return argmedian


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
        This argument allows us to specify an offet from the diagonal
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


def randomly_flip(flips, max_flips, rng=None):
    """Randomly flips some channels.

    Parameters
    ----------
    flips : numpy.ndarray
        Vector of 1s and -1s indicating which channels to flip.
    max_flips : int
        Maximum number of channels to change in this function.
    rng : numpy.random.Generator
        Random generator used to draw the flips. If ``None``, an unseeded generator is
        created, so the result is not reproducible; pass a seeded generator (e.g. from
        ``numpy.random.default_rng(seed)``) for reproducibility.

    Returns
    -------
    new_flips : numpy.ndarray
        Vector of 1s and -1s indicating which channels to flip.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Select the number of channels to flip
    n_channels_to_flip = rng.integers(1, max_flips + 1)

    # Select the channels to flip
    n_channels = flips.shape[0]
    random_channels_to_flip = rng.choice(n_channels, size=n_channels_to_flip, replace=False)
    new_flips = np.copy(flips)
    new_flips[random_channels_to_flip] *= -1

    return new_flips


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


def apply_flips(outdir, subject, flips, epoched=False, source_method="lcmv"):
    """Saves the sign flipped data.

    Parameters
    ----------
    outdir : str
        Path to source reconstruction directory.
    subject : str
        Subject name/id.
    flips : numpy.ndarray
        Flips to apply.
    epoched : bool
        Are we performing sign flipping on parc-raw.fif (epoched=False)
        or parc-epo.fif files (epoched=True)?
    source_method : str, optional
        Which parcellation file should we apply flips to.
    """
    if epoched:
        parc_file = op.join(outdir, str(subject), "parc", "parc-epo.fif")
        epochs = mne.read_epochs(parc_file, verbose=False)
        sflip_epochs = epochs.copy()
        sflip_epochs.load_data()

        # Flip the sign of the channels
        def flip(data):
            return data * flips[np.newaxis, :, np.newaxis]

        sflip_epochs.apply_function(flip, picks=_get_parc_chans(epochs), channel_wise=False)

        # Save
        outfile = op.join(outdir, str(subject), str(subject) + f"_sflip_{source_method}-parc-epo.fif")
        log_or_print(f"saving: {outfile}")
        sflip_epochs.save(outfile, overwrite=True)

    else:
        # Load parcellated data
        parc_file = op.join(outdir, str(subject), "parc", f"{source_method}-parc-raw.fif")
        raw = mne.io.read_raw_fif(parc_file, verbose=False)
        sflip_raw = raw.copy()
        sflip_raw.load_data()

        # Flip the sign of the channels
        def flip(data):
            return data * flips[:, np.newaxis]

        sflip_raw.apply_function(flip, picks=_get_parc_chans(raw), channel_wise=False)

        # Save
        outfile = op.join(outdir, str(subject), str(subject) + f"_sflip_{source_method}-parc-raw.fif")
        log_or_print(f"saving: {outfile}")
        sflip_raw.save(outfile, overwrite=True)


def time_embed(x, n_embeddings):
    """Performs time-delay embedding.

    Parameters
    ----------
    x : numpy.ndarray
        Time series data. Shape must be (n_samples, n_channels).
    n_embeddings : int
        Number of samples in which to shift the data. Must be an odd number.

    Returns
    -------
    sliding_window_view
        Time embedded data. Shape is (n_samples, n_channels * n_embeddings).
    """
    if n_embeddings % 2 == 0:
        raise ValueError("n_embeddings must be an odd number.")

    te_shape = (x.shape[0] - (n_embeddings - 1), x.shape[1] * n_embeddings)

    return np.lib.stride_tricks.sliding_window_view(x=x, window_shape=te_shape[0], axis=0).T[..., ::-1].reshape(te_shape)


def std_data(x):
    """Standardizes (z-transforms) the data.

    Parameters
    ----------
    x : numpy.ndarray
        Data. Shape must be (n_samples, n_channels).

    Returns
    -------
    std_x: numpy.ndarray
        Standardized time series.
    """
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)
