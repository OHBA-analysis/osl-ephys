"""The LCMV beamformer: filters from the forward model and the data covariance, their application, virtual electrodes, weights and leadfields."""

from __future__ import annotations

import os
import numpy as np
import mne
from mne.beamformer._compute_beamformer import (
    Beamformer,
    _compute_beamformer,
    _prepare_beamformer_input,
    _proj_whiten_data,
)
from mne.minimum_norm.inverse import _check_depth, _check_reference
from mne.utils import logger as mne_logger, warn
from osl_ephys.session import Session, check_up_to_date
from osl_ephys.rhino import forward as rhino_forward
from osl_ephys.source_recon.beamforming.bilateral import _find_bilateral_pairs
from osl_ephys.source_recon.beamforming.grid import (
    _get_gridstep,
    _get_mni_grid,
    _get_source_coords_mni,
)


def lcmv_beamformer(
    session: Session,
    data: str | mne.io.Raw | mne.Epochs | None = None,
    chantypes: str | list[str] | None = None,
    data_cov: mne.Covariance | None = None,
    noise_cov: mne.Covariance | None = None,
    pick_ori: str | None = "max-power-pre-weight-norm",
    rank: str | dict = "info",
    noise_rank: str | dict = "info",
    reduce_rank: bool = True,
    frequency_range: list[float] | None = None,
    use_bilateral_pairs: bool = False,
    bilateral_tol: float | None = None,
    bilateral_tol_midline: float | None = None,
    **kwargs,
) -> None:
    """Compute LCMV spatial filter.

    The filters are saved to session.filters_file.

    Parameters
    ----------
    session : Session
        Files of the session.
    data : str | instance of mne.Raw | mne.Epochs, optional
        The measurement data to specify the channels to include. Bad channels
        in info['bads'] are not used. Will also be used to calculate data_cov.
        If None, session.preproc_file is used.
    chantypes : list or str
        List of channel types to use to calculate the noise covariance.
        E.g. ['eeg'], ['mag', 'grad'], ['eeg', 'mag', 'grad'].
    data_cov : instance of mne.Covariance | None
        The data covariance. If None, it is computed from data.
    noise_cov : instance of mne.Covariance | None
        The noise covariance matrix used to whiten.
        If None will be computed from data as a diagonal matrix with variances
        set to the average of all sensors of that type.
    pick_ori : None | 'normal' | 'max-power' | 'vector' | 'max-power-pre-weight-norm'
        The source orientation to compute the beamformer in. 'max-power'
        picks the orientation that maximizes the power after weight
        normalization (see mne.beamformer.make_lcmv).
        'max-power-pre-weight-norm' picks the orientation that maximizes the
        power of the unit-gain beamformer, (l^T C^-1 l)^-1, i.e. before weight
        normalization; use reduce_rank=True with this option for MEG.
    rank : dict
        Calculate the rank only for a subset of channel types, and
        explicitly specify the rank for the remaining channel types.
        This can be extremely useful if you already know the rank of (part
        of) your data, for instance in case you have calculated it earlier.
        This parameter must be a dictionary whose keys correspond to channel
        types in the data (e.g. 'meg', 'mag', 'grad', 'eeg'), and whose
        values are integers representing the respective ranks. For example,
        {'mag': 90, 'eeg': 45} will assume a rank of 90 and 45 for
        magnetometer data and EEG data, respectively.
        The ranks for all channel types present in the data, but not
        specified in the dictionary will be estimated empirically. That is,
        if you passed a dataset containing magnetometer, gradiometer, and
        EEG data together with the dictionary from the previous example,
        only the gradiometer rank would be determined, while the specified
        magnetometer and EEG ranks would be taken for granted.
    noise_rank : dict | None | 'full' | 'info'
        This controls the rank computation that can be read from the measurement
        info or estimated from the data. When a noise covariance is used for
        whitening, this should reflect the rank of that covariance, otherwise
        amplification of noise components can occur in whitening (e.g., often
        during source localization).
    reduce_rank : bool
        Whether to reduce the rank by one during computation of the filter.
        This removes the orientation with the weakest lead field at each
        dipole (the radial orientation for MEG).
    frequency_range : list
        Lower and upper range (in Hz) for frequency range to bandpass filter.
        If None, no filtering is performed.
    use_bilateral_pairs : bool
        Should we compute joint beamformer weights for pairs of bilaterally
        symmetric dipoles? This avoids the suppression of correlated bilateral
        sources (e.g. left and right auditory cortex responding to the same
        stimulus) that occurs with a standard LCMV beamformer. Requires
        pick_ori='max-power' or 'max-power-pre-weight-norm'.
    bilateral_tol : float
        Tolerance (in mm) for pairing dipoles mirrored across the midline:
        the mean absolute difference between the x, y and z coordinates of a
        dipole and the mirror image of its partner. If None, half the gridstep
        of the dipole grid is used, which pairs a dipole with its mirror image
        or, if there is no dipole there, with a dipole one gridstep from it.
        Recommended value: gridstep / 2.
    bilateral_tol_midline : float
        Dipoles closer (in mm) than this to the midline (x = 0 in MNI space)
        are not paired. If None, bilateral_tol is used.
    **kwargs : keyword arguments
        reg, weight_norm, depth and inversion, see mne.beamformer.make_lcmv.
    """
    print("")
    print("Making LCMV beamformer")
    print("----------------------")

    check_up_to_date(session, "forward model")
    if data is None:
        data = session.preproc_file

    if chantypes is None:
        raise ValueError("chantypes must be passed.")

    if isinstance(chantypes, str):
        chantypes = [chantypes]

    # Load data
    if isinstance(data, str):
        if "epo.fif" in data:
            data = mne.read_epochs(data)
        else:
            data = mne.io.read_raw_fif(data, preload=True)

    # Bandpass filter
    if frequency_range is not None:
        data.filter(
            l_freq=frequency_range[0],
            h_freq=frequency_range[1],
            method="iir",
            iir_params={"order": 5, "ftype": "butter"},
        )

    # Load forward solution
    fwd = mne.read_forward_solution(session.fwd_model_file)

    data_cov, noise_cov = _data_and_noise_covariance(
        data, chantypes, rank, data_cov, noise_cov
    )

    if (
        pick_ori == "max-power-pre-weight-norm"
        and not reduce_rank
        and any(c in ("meg", "mag", "grad") for c in chantypes)
    ):
        warn(
            "pick_ori='max-power-pre-weight-norm' with reduce_rank=False can "
            "pick orientations with a weak MEG lead field (e.g. radial). Use "
            "reduce_rank=True for MEG."
        )

    # Find pairs of bilaterally symmetric dipoles to beamform jointly. The
    # other dipoles (unpaired or on the midline) get standard weights.
    multi_dipoles = None
    if use_bilateral_pairs:
        if pick_ori not in ("max-power", "max-power-pre-weight-norm"):
            raise ValueError(
                "use_bilateral_pairs=True requires pick_ori='max-power' or "
                "'max-power-pre-weight-norm'."
            )
        _, multi_dipoles, _, _ = _find_bilateral_pairs(
            session, fwd, bilateral_tol, bilateral_tol_midline
        )
        if len(multi_dipoles) == 0:
            warn(
                "No bilateral pairs found, using a standard beamformer. "
                "Consider increasing bilateral_tol."
            )
            multi_dipoles = None

    # Make filters
    lcmv_params = dict(
        pick_ori=pick_ori,
        rank=rank,
        noise_rank=noise_rank,
        reduce_rank=reduce_rank,
        **{k: v for k, v in kwargs.items() if k != "verbose"},
    )
    filters = _make_lcmv(
        data.info,
        fwd,
        data_cov,
        noise_cov=noise_cov,
        multi_dipoles=multi_dipoles,
        **lcmv_params,
    )

    # Keep the settings so we can compute filters for other locations,
    # see virtual_electrodes
    filters["osl_lcmv_params"] = lcmv_params
    filters["osl_bilateral_pairs"] = dict(
        used=multi_dipoles is not None,
        bilateral_tol=bilateral_tol,
        bilateral_tol_midline=bilateral_tol_midline,
    )

    os.makedirs(os.path.dirname(session.filters_file), exist_ok=True)
    print(f"Saving {session.filters_file}")
    filters.save(session.filters_file, overwrite=True)

    print("LCMV beamformer complete.")


def apply_lcmv_beamformer(
    session: Session,
    raw: mne.io.Raw | mne.Epochs | None = None,
    reject_by_annotation: str | None = "omit",
    reference_brain: str = "mni",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an LCMV beamformer.

    Parameters
    ----------
    session : Session
        Files of the session.
    raw : instance of mne.io.Raw or mne.Epochs, optional
        The data to apply the LCMV filter to.
        If None, session.preproc_file is used.
    reject_by_annotation : str | None
        How to treat the samples of a Raw object in annotations starting with
        'bad': 'omit' drops them, None keeps them.
    reference_brain : str, optional
        Either 'head' or 'mni'. If 'mni', the data is returned for each voxel
        of the MNI grid of the forward model (session.mni_grid_file). Voxels without a
        dipole (outside the inner skull) are zero.

    Returns
    -------
    voxel_data : np.ndarray
        Voxel data in the space specified by the 'reference_brain' argument.
        Shape is (voxels, time) for a Raw object or (voxels, time, epochs) for
        an Epochs object, with an orientation axis of length 3 after voxels
        for pick_ori='vector'.
    coords : np.ndarray
        Coordinates (in mm) for each voxel in the reference brain space.
        Shape is (voxels, 3).
    """
    print()
    print("Applying LCMV beamformer")
    print("------------------------")

    check_up_to_date(session, "LCMV filters")
    data, filters, epochs_shape = _get_filter_input_data(
        session, raw, reject_by_annotation
    )
    voxel_data_head = filters["weights"] @ data
    if filters["is_free_ori"]:
        # (dipoles, orientations, samples). Unless pick_ori='vector', combine
        # the orientations as mne.beamformer.apply_lcmv does.
        voxel_data_head = voxel_data_head.reshape(-1, 3, data.shape[-1])
        if filters["pick_ori"] != "vector":
            voxel_data_head = np.sqrt(np.sum(voxel_data_head**2, axis=1))
    if epochs_shape is not None:
        voxel_data_head = voxel_data_head.reshape(
            *voxel_data_head.shape[:-1], *epochs_shape
        )

    fwd = mne.read_forward_solution(session.fwd_model_file)
    vs = fwd["src"][0]
    if reference_brain == "head":
        # Get coordinates in head space
        voxel_data = voxel_data_head
        coords = vs["rr"][vs["vertno"]] * 1000  # in mm
    else:
        # The dipoles are at the voxels of an MNI grid
        coords = _get_mni_grid(session, fwd)
        voxel_data = np.zeros((len(coords),) + voxel_data_head.shape[1:])
        voxel_data[vs["vertno"]] = voxel_data_head

    print("Applying LCMV beamformer complete.")

    return voxel_data, coords


def virtual_electrodes(
    session: Session,
    coords: list | np.ndarray,
    raw: mne.io.Raw | mne.Epochs | None = None,
    reject_by_annotation: str | None = "omit",
    use_bilateral_pairs: bool | None = None,
) -> np.ndarray:
    """Beamform the activity at MNI coordinates (virtual electrodes).

    The lead field is computed for a dipole at the exact location of each
    coordinate and the LCMV beamformer weights are computed with the same data
    covariance and settings used by :func:`lcmv_beamformer`, so the time
    course is not taken from the nearest dipole of the MNI grid.

    Parameters
    ----------
    session : Session
        Files of the session. :func:`lcmv_beamformer` must have been
        run.
    coords : list | np.ndarray
        MNI coordinates (in mm). Shape is (3,) for a single coordinate or
        (n_coords, 3). The coordinates must be inside the inner skull.
    raw : mne.io.Raw or mne.Epochs, optional
        The data to beamform. If None, session.preproc_file is used.
    reject_by_annotation : str | None
        How to treat the samples of a Raw object in annotations starting with
        'bad': 'omit' drops them, None keeps them.
    use_bilateral_pairs : bool, optional
        Should we compute joint beamformer weights with the coordinate
        mirrored across the midline (x = 0)? Coordinates close to the midline,
        or whose mirror image is outside the inner skull, use a single dipole.
        If None, we do this if the filters were computed with
        :code:`use_bilateral_pairs=True`. Note the dipoles of the MNI grid are
        paired across the grid's own plane of symmetry, which can be up to half
        a voxel from x = 0.

    Returns
    -------
    data : np.ndarray
        Time course at each coordinate. Shape is (n_coords, time) or
        (n_coords, time, epochs), without the first axis if a single
        coordinate is passed.
    """
    print()
    print("Calculating virtual electrodes")
    print("------------------------------")

    coords = np.asarray(coords, dtype=float)
    single = coords.ndim == 1
    coords = np.atleast_2d(coords)
    n_coords = len(coords)

    check_up_to_date(session, "LCMV filters")
    if raw is None:
        raw = mne.io.read_raw_fif(session.preproc_file, preload=True)

    # Settings used to compute the filters on the MNI grid
    filters = mne.beamformer.read_beamformer(session.filters_file)
    lcmv_params = filters["osl_lcmv_params"]
    lcmv_params["reduce_rank"] = bool(lcmv_params["reduce_rank"])  # h5io int
    bilateral = filters["osl_bilateral_pairs"]
    if lcmv_params["pick_ori"] in [None, "vector"]:
        raise ValueError("virtual_electrodes requires a scalar beamformer.")
    if use_bilateral_pairs is None:
        use_bilateral_pairs = bool(bilateral["used"])

    # Sensors used by the forward model
    fwd = mne.read_forward_solution(session.fwd_model_file, verbose=False)
    meg = len(mne.pick_types(fwd["info"], meg=True, ref_meg=False)) > 0
    eeg = len(mne.pick_types(fwd["info"], meg=False, eeg=True)) > 0

    # Dipole locations: the coordinates, then (for bilateral pairs) their
    # mirror images across the midline
    positions = coords
    if use_bilateral_pairs:
        tol_midline = bilateral["bilateral_tol_midline"]
        if tol_midline is None:
            tol_midline = bilateral["bilateral_tol"]
        if tol_midline is None:
            tol_midline = _get_gridstep(_get_mni_grid(session, fwd) / 1000) / 2
        paired = np.flatnonzero(np.abs(coords[:, 0]) >= tol_midline)
        positions = np.concatenate([coords, coords[paired] * [-1, 1, 1]])

    # Lead fields at the exact locations
    ve_fwd, inside = _forward_model_at_coords(session, positions, meg=meg, eeg=eeg)
    if not inside[:n_coords].all():
        raise ValueError(
            "The following coordinates are outside the inner skull: "
            f"{coords[~inside[:n_coords]].tolist()}"
        )

    multi_dipoles = None
    if use_bilateral_pairs:
        # A coordinate whose mirror image is outside the inner skull gets a
        # single dipole
        dipole = np.cumsum(inside) - 1  # index of each position in ve_fwd
        has_mirror = inside[n_coords:]
        multi_dipoles = [
            [i, dipole[n_coords + j]] for j, i in enumerate(paired) if has_mirror[j]
        ]
        print(f"Using bilateral pairs for {len(multi_dipoles)} coordinate(s)")
        if not multi_dipoles:
            multi_dipoles = None

    # Beamformer weights with the settings used for the MNI grid
    info = mne.pick_info(
        raw.info, mne.pick_channels(raw.ch_names, filters["ch_names"], ordered=False)
    )
    ve_filters = _make_lcmv(
        info,
        ve_fwd,
        filters["data_cov"],
        noise_cov=filters["noise_cov"],
        multi_dipoles=multi_dipoles,
        **lcmv_params,
    )

    # Apply to the data
    data, _, epochs_shape = _get_filter_input_data(
        session, raw, reject_by_annotation, filters=ve_filters
    )
    ve_data = ve_filters["weights"][:n_coords] @ data
    if epochs_shape is not None:
        ve_data = ve_data.reshape(n_coords, *epochs_shape)

    print("Virtual electrodes complete.")

    return ve_data[0] if single else ve_data


def lcmv_weights(session: Session) -> tuple[np.ndarray, np.ndarray]:
    """The LCMV weights of a session, on the MNI grid.

    Parameters
    ----------
    session : Session
        Files of the session; the filters come from ``session.filters_file``.

    Returns
    -------
    weights : (n_dipoles, n_channels) numpy.ndarray
        Beamformer weights, applied to whitened data when the filters whiten.
    coords : (n_dipoles, 3) numpy.ndarray
        MNI coordinates (mm) of the dipoles.
    """
    filters = mne.beamformer.read_beamformer(session.filters_file)
    weights = filters["weights"]
    if filters.get("whitener") is not None:
        weights = weights @ filters["whitener"]
    fwd = mne.read_forward_solution(session.fwd_model_file, verbose=False)
    return weights, _get_source_coords_mni(session, fwd)


def leadfields(
    session: Session, orientation: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """The leadfields of a session's forward model, on the MNI grid.

    Parameters
    ----------
    session : Session
        Files of the session; the forward model comes from
        ``session.fwd_model_file``.
    orientation : str, optional
        How the three dipole orientations are reduced to one: ``'l2-norm'``
        (the norm across x, y, z), ``'max-dim'`` (the largest of the three),
        ``'max-power'`` (projected onto the orientation of maximum power), or
        None for no reduction.

    Returns
    -------
    leadfields : (n_channels, n_dipoles) or (n_channels, n_dipoles, 3) numpy.ndarray
        Leadfields, or (n_channels, 3, n_dipoles) flattened to
        (n_channels * 3, n_dipoles) when not reduced.
    coords : (n_dipoles, 3) numpy.ndarray
        MNI coordinates (mm) of the dipoles.
    """
    fwd = mne.read_forward_solution(session.fwd_model_file, verbose=False)
    L = fwd["sol"]["data"]
    L = L.reshape((L.shape[0], -1, 3))
    if orientation == "l2-norm":
        out = np.linalg.norm(L, axis=-1)
    elif orientation == "max-dim":
        L_max, L_min = np.max(L, axis=-1), np.min(L, axis=-1)
        L_max[abs(L_min) > L_max] = L_min[abs(L_min) > L_max]
        out = L_max
    elif orientation == "max-power":
        # The eigenvector with the smallest eigenvalue of L^T L
        LT_L = np.matmul(L.transpose(1, 2, 0), L.transpose((1, 0, 2)))
        eig_vals, eig_vecs = np.linalg.eig(LT_L)
        order = np.argsort(np.abs(eig_vals), axis=-1)
        max_power_ori = eig_vecs[np.arange(len(eig_vecs)), :, order[:, 0]]
        signs = np.sign(
            np.sum(max_power_ori * fwd["source_nn"][2::3], axis=1, keepdims=True)
        )
        signs[signs == 0] = 1.0
        max_power_ori *= signs
        out = np.sum(L * max_power_ori[None, ...], axis=-1)
    elif orientation is None:
        out = np.transpose(L, (0, 2, 1)).reshape(-1, L.shape[1])
    else:
        raise ValueError(
            "orientation must be 'l2-norm', 'max-dim', 'max-power' or None."
        )
    return out, _get_source_coords_mni(session, fwd)


def _data_and_noise_covariance(
    data: mne.io.Raw | mne.Epochs,
    chantypes: list[str],
    rank: str | dict,
    data_cov: mne.Covariance | None = None,
    noise_cov: mne.Covariance | None = None,
) -> tuple[mne.Covariance, mne.Covariance]:
    """Data and noise covariance for the LCMV beamformer.

    Parameters
    ----------
    data : mne.io.Raw | mne.Epochs
        Sensor data.
    chantypes : list of str
        Channel types in the data, e.g. ['mag', 'grad'].
    rank : str | dict
        Rank of the data, see :func:`lcmv_beamformer`.
    data_cov : mne.Covariance, optional
        Data covariance. If None, it is computed from the data.
    noise_cov : mne.Covariance, optional
        Noise covariance. If None, a diagonal matrix with the variance of each
        channel type set to the mean variance of its channels.

    Returns
    -------
    data_cov : mne.Covariance
        Data covariance.
    noise_cov : mne.Covariance
        Noise covariance.
    """
    if data_cov is None:
        # Note that if chantypes are meg, eeg; and meg includes mag, grad then
        # compute_covariance will project data separately for meg and eeg to
        # reduced rank subspace (i.e. mag and grad will be combined together
        # inside the meg subspace, eeg will haeve a separate subspace). I.e.
        # data will become (ntpts x (rank_meg + rank_eeg)) and cov will be
        # (rank_meg + rank_eeg) x (rank_meg + rank_eeg) and include correlations
        # between eeg and meg subspaces. The output data_cov is cov projected
        # back onto the individual sensor types mag, grad, eeg.
        #
        # Prior to computing anything, including the subspaces each of mag, grad,
        # eeg are scaled so that they are on comparable scales to aid mixing in
        # the subspace and improve numerical stability. Note that in the
        # output data_cov the scalings have been undone.
        if isinstance(data, mne.BaseEpochs):
            data_cov = mne.compute_covariance(data, method="empirical", rank=rank)
        else:
            data_cov = mne.compute_raw_covariance(data, method="empirical", rank=rank)

    if noise_cov is None:
        # Calculate noise covariance matrix
        #
        # Later this will be inverted and used to whiten the data AND the lead
        # fields as part of the source recon. See:
        #
        #   https://www.sciencedirect.com/science/article/pii/S1053811914010325
        #
        # In MNE, the noise cov is normally obtained from empty room noise
        # recordings or from a baseline period. Here (if no noise cov is passed
        # in) we compute a diagonal noise cov with the variances set to the mean
        # variance of each sensor type (e.g. mag, grad, eeg).
        n_channels = data_cov.data.shape[0]
        noise_cov_diag = np.zeros(n_channels)
        for chantype in chantypes:
            # Indices of this channel type
            type_data = data.copy().pick(chantype, exclude="bads")
            inds = [data_cov.ch_names.index(c) for c in type_data.info["ch_names"]]

            # Mean variance of channels of this type
            variance = np.mean(np.diag(data_cov.data)[inds])
            noise_cov_diag[inds] = variance
            print(f"Variance for chantype {chantype} is {variance}")

        bads = [b for b in data.info["bads"] if b in data_cov.ch_names]
        noise_cov = mne.Covariance(
            noise_cov_diag,
            data_cov.ch_names,
            bads,
            data.info["projs"],
            nfree=1e10,
        )
    return data_cov, noise_cov


def _make_lcmv(
    info: mne.Info,
    fwd: mne.Forward,
    data_cov: mne.Covariance,
    reg: float = 0.05,
    noise_cov: mne.Covariance | None = None,
    pick_ori: str | None = None,
    rank: str | dict = "info",
    noise_rank: str | dict = "info",
    weight_norm: str | None = "unit-noise-gain-invariant",
    reduce_rank: bool = False,
    depth: float | dict | None = None,
    inversion: str = "matrix",
    multi_dipoles: list[list[int]] | None = None,
) -> Beamformer:
    """Compute LCMV spatial filter.

    Same as mne.beamformer.make_lcmv (and uses its functions), plus:

    - noise_rank: the rank of the noise covariance, separate from the rank of
      the data covariance.
    - pick_ori='max-power-pre-weight-norm'.
    - multi_dipoles: joint weights for groups of dipoles.

    Parameters
    ----------
    info : mne.Info
        Measurement info, specifies the channels to use.
    fwd : mne.Forward
        Forward solution.
    data_cov : mne.Covariance
        Data covariance.
    reg : float
        Regularization of the whitened data covariance.
    noise_cov : mne.Covariance
        Noise covariance, used to whiten.
    pick_ori : None | 'normal' | 'max-power' | 'vector' | 'max-power-pre-weight-norm'
        Source orientation, see :func:`lcmv_beamformer`.
    rank : str | dict
        Rank of the data covariance, see :func:`lcmv_beamformer`.
    noise_rank : str | dict
        Rank of the noise covariance, see :func:`lcmv_beamformer`.
    weight_norm : None | 'unit-noise-gain-invariant' | 'unit-noise-gain' | 'nai'
        Weight normalization.
    reduce_rank : bool
        Reduce the rank of the lead field of each dipole by one?
    depth : None | float | dict
        Depth weighting of the forward solution.
    inversion : 'matrix' | 'single'
        Inversion scheme for the denominator of the weights.
    multi_dipoles : list of list of int
        Groups of dipoles (indices of the in-use sources of the forward model)
        to compute joint weights for, e.g. bilateral pairs. If None, a
        standard beamformer is computed. Other dipoles get standard weights.

    Returns
    -------
    filters : mne.beamformer.Beamformer
        LCMV filters.
    """
    mne.utils._check_option(
        "pick_ori",
        pick_ori,
        ("normal", "max-power", "vector", "max-power-pre-weight-norm", None),
    )
    mne.utils._check_option(
        "weight_norm",
        weight_norm,
        ["unit-noise-gain-invariant", "unit-noise-gain", "nai", None],
    )
    if pick_ori is not None and mne.forward.is_fixed_orient(fwd):
        raise ValueError(
            f"pick_ori={pick_ori!r} requires a free orientation forward model."
        )
    if multi_dipoles is not None:
        if pick_ori not in ("max-power", "max-power-pre-weight-norm"):
            raise ValueError(
                "multi_dipoles requires pick_ori='max-power' or "
                "'max-power-pre-weight-norm'."
            )

    # 1. Channels and ranks. rank='info' uses the MaxFilter rank (from
    # proc_history) for the data covariance, but not for a diagonal noise
    # covariance.
    info = mne._fiff.meas_info._simplify_info(info, keep=("proc_history",))
    noise_cov, _, _ = mne.utils._check_one_ch_type(
        "lcmv", info, fwd, data_cov, noise_cov
    )
    info = mne.pick_info(
        info, mne.utils._check_info_inv(info, fwd, data_cov, noise_cov)
    )
    data_rank = mne.compute_rank(data_cov, rank=rank, info=info)
    data_rank_total = sum(data_rank.values())
    noise_info = mne._fiff.meas_info._simplify_info(info) if noise_cov["diag"] else info
    noise_rank = mne.compute_rank(noise_cov, rank=noise_rank, info=noise_info)
    mne_logger.info(f"Making LCMV beamformer with data cov rank {data_rank}")
    mne_logger.info(f"Making LCMV beamformer with noise cov rank {noise_rank}")

    # 2. What MNE computes and what we compute:
    #
    # - 'max-power-pre-weight-norm': MNE's 'max-power' orientation and weights
    #   for the unit-gain beamformer (weight_norm=None), normalized in step 6.
    # - multi_dipoles: MNE's weights for each dipole on its own, replaced by
    #   joint weights in step 5 and normalized in step 6.
    pre_weight_norm = pick_ori == "max-power-pre-weight-norm"
    mne_pick_ori = "max-power" if pre_weight_norm else pick_ori
    mne_weight_norm = None if pre_weight_norm else weight_norm
    normalize_weights = weight_norm is not None and (
        pre_weight_norm or multi_dipoles is not None
    )

    # 3. Whitened lead field
    depth = _check_depth(depth, "depth_sparse")
    if inversion == "single":
        depth["combine_xyz"] = False
    is_free_ori, info, proj, vertno, G, whitener, nn, orient_std = (
        _prepare_beamformer_input(
            info,
            fwd,
            None,
            mne_pick_ori,
            noise_cov=noise_cov,
            rank=noise_rank,
            pca=False,
            **depth,
        )
    )
    ch_names = list(info["ch_names"])
    data_cov = mne.pick_channels_cov(data_cov, include=ch_names)
    data_cov.pop("estimator", None)
    Cm = data_cov._get_square()

    # 4. Weights and orientations
    W, max_power_ori = _compute_beamformer(
        G,
        Cm,
        reg,
        3 if is_free_ori else 1,
        mne_weight_norm,
        mne_pick_ori,
        reduce_rank,
        data_rank_total,
        inversion=inversion,
        nn=nn,
        orient_std=orient_std,
        whitener=whitener,
    )

    # 5. Joint unit-gain weights for the multi-dipoles, (L^T C^-1 L)^-1 L^T C^-1.
    # MNE's weights for each dipole are W = D L^T C^-1 (D diagonal), so the
    # joint weights are (W L)^-1 W, with L the lead field at the max-power
    # orientation. With the pseudo-inverse, a dipole paired with itself gets
    # its single-dipole weights.
    if multi_dipoles is not None:
        L = np.einsum("cdo,do->cd", G.reshape(len(G), -1, 3), max_power_ori)
        for dipoles in multi_dipoles:
            W[dipoles] = (
                np.linalg.pinv(W[dipoles] @ L[:, dipoles], rcond=1e-7) @ W[dipoles]
            )

    # 6. Weight normalization. The beamformer is scalar, so 'unit-noise-gain'
    # (and 'unit-noise-gain-invariant') scales the weights of each dipole to
    # unit norm, and 'nai' additionally divides by sqrt(noise).
    if normalize_weights:
        norm = np.linalg.norm(W, axis=1, keepdims=True)
        norm[norm == 0] = np.inf
        W = W / norm
        if weight_norm == "nai":
            # MNE's 'nai' weights are the unit-noise-gain weights divided by
            # sqrt(noise), where the noise level (the same for every dipole)
            # is estimated from the data covariance. Take the factor from
            # MNE's 'nai' weights for one dipole.
            i = np.argmax(np.linalg.norm(G, axis=0))
            W_nai, _ = _compute_beamformer(
                G[:, i : i + 1],
                Cm,
                reg,
                1,
                "nai",
                None,
                False,
                data_rank_total,
                inversion="matrix",
                nn=nn[:1],
                orient_std=np.ones(1),
                whitener=whitener,
            )
            W *= np.linalg.norm(W_nai)

    return Beamformer(
        kind="LCMV",
        weights=W,
        data_cov=data_cov,
        noise_cov=noise_cov,
        whitener=whitener,
        weight_norm=weight_norm,
        pick_ori=pick_ori,
        ch_names=ch_names,
        proj=proj,
        is_ssp=bool(info["projs"]),
        vertices=vertno,
        # Scalar beamformer unless pick_ori is None or 'vector'
        is_free_ori=is_free_ori and pick_ori in (None, "vector"),
        n_sources=fwd["nsource"],
        src_type=mne.source_estimate._get_src_type(fwd["src"], vertno),
        source_nn=fwd["source_nn"].copy(),
        subject=mne.forward._subject_from_forward(fwd),
        rank=data_rank_total,
        max_power_ori=max_power_ori,
        inversion=inversion,
    )


def _get_filter_input_data(
    session: Session,
    raw: mne.io.Raw | mne.Epochs | None,
    reject_by_annotation: str | None,
    filters: Beamformer | None = None,
) -> tuple[np.ndarray, Beamformer, tuple | None]:
    """Get the sensor data that the LCMV weights are applied to.

    Parameters
    ----------
    session : Session
        Files of the session.
    raw : mne.io.Raw or mne.Epochs
        Data. If None, session.preproc_file is used.
    reject_by_annotation : str | None
        How to treat the samples of a Raw object in annotations starting with
        'bad': 'omit' drops them, None keeps them.
    filters : Beamformer, optional
        LCMV filters. If None, session.filters_file is used.

    Returns
    -------
    data : np.ndarray
        (channels, samples) data after SSP projection and whitening. For
        Epochs, the samples of each epoch are concatenated.
    filters : Beamformer
        LCMV filters.
    epochs_shape : tuple
        (time, epochs) shape of the samples for Epochs, None for Raw.
    """
    if raw is None:
        raw = mne.io.read_raw_fif(session.preproc_file, preload=True)
    if filters is None:
        filters = mne.beamformer.read_beamformer(session.filters_file)

    _check_reference(raw, filters["ch_names"])
    chan_inds = mne.utils._check_channels_spatial_filter(raw.ch_names, filters)

    if isinstance(raw, mne.BaseEpochs):
        data = raw.get_data(picks=chan_inds)  # (epochs, channels, time)
        epochs_shape = (data.shape[2], data.shape[0])
        data = np.transpose(data, (1, 2, 0)).reshape(data.shape[1], -1)
    else:
        data = raw.get_data(picks=chan_inds, reject_by_annotation=reject_by_annotation)
        epochs_shape = None

    projs = mne.pick_info(raw.info, chan_inds)["projs"]
    data = _proj_whiten_data(data, projs, filters)
    return data, filters, epochs_shape


def _forward_model_at_coords(
    session: Session, coords_mni: np.ndarray, meg: bool = True, eeg: bool = False
) -> tuple[mne.Forward, np.ndarray]:
    """Compute the forward model for dipoles at MNI coordinates.

    Parameters
    ----------
    session : Session
        Files of the session.
    coords_mni : np.ndarray
        (n, 3) coordinates in MNI space in mm.
    meg : bool, optional
        Include MEG sensors?
    eeg : bool, optional
        Include EEG sensors?

    Returns
    -------
    fwd : mne.Forward
        Forward model with one (free orientation) dipole per coordinate inside
        the inner skull, in the same order as coords_mni. None if no
        coordinate is inside.
    inside : np.ndarray
        (n,) whether each coordinate is inside the inner skull.
    """
    src = rhino_forward._mni_source_space(session, coords_mni)
    info, head_mri_t = rhino_forward._read_head_model(session)
    try:
        fwd = rhino_forward._make_fwd_solution(
            info,
            head_mri_t,
            src=src,
            bem=session.bem_solution_file,
            meg=meg,
            eeg=eeg,
            ignore_ref=True,
            verbose=False,
        )
    except RuntimeError as e:
        if "No points left" not in str(e):
            raise
        return None, np.zeros(len(coords_mni), dtype=bool)
    return fwd, fwd["src"][0]["inuse"].astype(bool)
