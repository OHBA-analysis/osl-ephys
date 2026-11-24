"""Surface-based source reconstruction with FreeSurfer surfaces and MNE.

The alternative to RHINO: the head surfaces come from FreeSurfer's
``recon-all`` (or MNE's ``fsaverage`` when there is no MRI), coregistration
uses MNE's :class:`~mne.coreg.Coregistration`, the forward model lives on a
cortical surface source space, and the inverse is a minimum-norm estimate.
Files go in ``{outdir}/freesurfer/{subject}`` (the FreeSurfer subject) and
``{outdir}/{id}/freesurfer`` (the session's coregistration, forward model and
inverse operator).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import mne
import numpy as np

from osl_ephys.session import Session, save_params

logger = logging.getLogger("osl_ephys")


class FreeSurferFilenames:
    """The FreeSurfer-route files of a session.

    Parameters
    ----------
    session : Session
        Files of the session. The FreeSurfer subject is its ``subject``
        attribute (the manifest's subject column), else its ``head_model_id``.
    """

    def __init__(self, session: Session):
        self.subjects_dir = Path(session.outdir) / "freesurfer"
        self.subject = getattr(session, "subject", session.head_model_id)
        self.subject_dir = self.subjects_dir / self.subject
        root = Path(session.outdir) / session.id / "freesurfer"
        self.root = root
        self.info_file = root / "info-raw.fif"
        self.trans_file = root / "coreg-trans.fif"
        self.coreg_plot_file = root / "coreg.png"
        self.coreg_quality_file = root / "coreg.json"
        self.source_space_file = root / "space-src.fif"
        self.morph_file = root / "space-src-morph.h5"
        self.fwd_model_file = root / "model-fwd.fif"
        self.fwd_model_params_file = root / "model-fwd.json"
        self.inverse_operator_file = root / "op-inv.fif"
        self.inverse_params_file = root / "op-inv.json"

    def fsaverage_source_space_file(self, spacing: str) -> Path:
        return self.subjects_dir / "fsaverage" / f"space-src-{spacing}.fif"


def setup_freesurfer(directory: str, subjects_dir: str | None = None) -> None:
    """Point the environment at a FreeSurfer installation.

    Parameters
    ----------
    directory : str
        FreeSurfer installation directory (``FREESURFER_HOME``).
    subjects_dir : str, optional
        ``SUBJECTS_DIR`` to use.
    """
    os.environ["FREESURFER_HOME"] = directory
    os.environ["PATH"] = f"{directory}/bin:{os.environ['PATH']}"
    if subjects_dir is not None:
        os.environ["SUBJECTS_DIR"] = subjects_dir


def check_freesurfer() -> None:
    """Raise if FreeSurfer is not available."""
    if "FREESURFER_HOME" not in os.environ or shutil.which("recon-all") is None:
        raise RuntimeError(
            "FreeSurfer was not found: run setup_freesurfer() or source its SetUpFreeSurfer.sh."
        )


def recon_all(session: Session, smri_file: str) -> None:
    """Run FreeSurfer's ``recon-all`` on the structural MRI of the subject.

    Parameters
    ----------
    session : Session
        Files of the session; the subject is ``session.head_model_id``.
    smri_file : str
        Structural MRI.
    """
    check_freesurfer()
    fs = FreeSurferFilenames(session)
    fs.subjects_dir.mkdir(parents=True, exist_ok=True)
    if fs.subject_dir.exists():
        logger.info(f"{fs.subject_dir} exists; recon-all is not run again.")
        return
    env = {**os.environ, "SUBJECTS_DIR": str(fs.subjects_dir)}
    subprocess.run(
        ["recon-all", "-i", str(smri_file), "-s", fs.subject, "-all"],
        check=True,
        env=env,
    )


def make_watershed_bem(session: Session, **kwargs) -> None:
    """FreeSurfer's watershed BEM surfaces for the subject (needs FreeSurfer)."""
    check_freesurfer()
    fs = FreeSurferFilenames(session)
    mne.bem.make_watershed_bem(
        subject=fs.subject, subjects_dir=str(fs.subjects_dir), **kwargs
    )


def fetch_fsaverage(session: Session) -> Path:
    """MNE's ``fsaverage`` in the session's FreeSurfer directory, downloaded if needed."""
    fs = FreeSurferFilenames(session)
    fs.subjects_dir.mkdir(parents=True, exist_ok=True)
    if not (fs.subjects_dir / "fsaverage" / "bem").exists():
        mne.datasets.fetch_fsaverage(subjects_dir=str(fs.subjects_dir), verbose=False)
    return fs.subjects_dir / "fsaverage"


def fsaverage_source_space(session: Session, spacing: str = "oct6") -> mne.SourceSpaces:
    """The fsaverage surface source space at a spacing, made once and saved."""
    fs = FreeSurferFilenames(session)
    path = fs.fsaverage_source_space_file(spacing)
    if path.exists():
        return mne.read_source_spaces(str(path), verbose=False)
    fetch_fsaverage(session)
    src = mne.setup_source_space(
        "fsaverage",
        spacing=spacing,
        subjects_dir=str(fs.subjects_dir),
        add_dist="patch",
        verbose=False,
    )
    mne.write_source_spaces(str(path), src, overwrite=True, verbose=False)
    return src


def _read_data(session: Session, data=None, frequency_range=None):
    """The session's preprocessed data (raw or epochs), optionally band-passed."""
    if data is None:
        if session.preproc_file.endswith("epo.fif"):
            data = mne.read_epochs(session.preproc_file, preload=True, verbose=False)
        else:
            data = mne.io.read_raw(session.preproc_file, preload=True, verbose=False)
    if frequency_range is not None:
        logger.info(f"Band-pass filtering {frequency_range[0]}-{frequency_range[1]} Hz")
        data = data.filter(
            l_freq=frequency_range[0],
            h_freq=frequency_range[1],
            method="iir",
            iir_params={"order": 5, "ftype": "butter"},
            verbose=False,
        )
    return data


def coregister(
    session: Session,
    allow_mri_scaling: bool = False,
    n_init: int = 20,
    fit_fiducials: dict | None = None,
    fit_icp: dict | None = None,
) -> None:
    """Coregister the sensors with the FreeSurfer head surface.

    Fiducials are estimated from fsaverage, then the digitisation is fitted
    to the head surface with ICP. Writes the transform, a plot and the
    distances in ``coreg.json``.

    Parameters
    ----------
    session : Session
        Files of the session.
    allow_mri_scaling : bool, optional
        Let the MRI be scaled to the head (for a template MRI).
    n_init : int, optional
        ICP iterations.
    fit_fiducials, fit_icp : dict, optional
        Options for :meth:`mne.coreg.Coregistration.fit_fiducials` and
        :meth:`~mne.coreg.Coregistration.fit_icp`.
    """
    fs = FreeSurferFilenames(session)
    fs.root.mkdir(parents=True, exist_ok=True)
    info = mne.io.read_info(session.preproc_file, verbose=False)
    mne.io.RawArray(np.zeros([len(info["ch_names"]), 1]), info, verbose=False).save(
        str(fs.info_file), overwrite=True, verbose=False
    )

    coreg = mne.coreg.Coregistration(
        info, fs.subject, str(fs.subjects_dir), fiducials="estimated"
    )
    if not allow_mri_scaling:
        coreg.set_scale_mode(None)
    coreg.fit_fiducials(**(fit_fiducials or {}), verbose=False)
    coreg.fit_icp(n_iterations=n_init, **(fit_icp or {}), verbose=False)
    mne.write_trans(str(fs.trans_file), coreg.trans, overwrite=True, verbose=False)

    dists = coreg.compute_dig_mri_distances() * 1e3  # mm
    fiducial_error = {
        name: float(
            np.linalg.norm(coreg._dig_dict[name] - coreg.fiducials.dig[i]["r"]) * 1e3
        )
        for i, name in enumerate(("lpa", "nasion", "rpa"))
    }
    quality = {
        "headshape_distance_mm": {
            "mean": float(dists.mean()),
            "min": float(dists.min()),
            "max": float(dists.max()),
        },
        "fiducial_error_mm": fiducial_error,
        "n_init": n_init,
    }
    save_params(str(fs.coreg_quality_file), quality)
    logger.info(
        f"Headshape to scalp distance: mean {dists.mean():.1f} mm, max {dists.max():.1f} mm"
    )
    _plot_coregistration(session, info, coreg.trans)


def _plot_coregistration(session: Session, info, trans) -> None:
    """Screenshot of the alignment; skipped with a warning if no renderer works."""
    fs = FreeSurferFilenames(session)
    try:
        import pyvista

        pyvista.OFF_SCREEN = True
        fig = mne.viz.plot_alignment(
            info,
            trans=trans,
            subject=fs.subject,
            subjects_dir=str(fs.subjects_dir),
            surfaces="head",
            dig=True,
            show_axes=True,
            verbose=False,
        )
        mne.viz.set_3d_view(fig, azimuth=90, elevation=90)
        fig.plotter.screenshot(str(fs.coreg_plot_file))
        fig.plotter.close()
    except Exception as error:
        warnings.warn(f"The coregistration plot could not be made: {error}")


def forward_model(
    session: Session,
    source_space: str = "surface",
    spacing: str | float = "oct6",
    model: str = "Single Layer",
    conductivity: tuple | None = None,
    ico: int = 4,
    mindist: float = 0.0,
    eeg: bool = False,
) -> None:
    """Forward model on the FreeSurfer surfaces.

    Parameters
    ----------
    session : Session
        Files of the session (coregistered with :func:`coregister`).
    source_space : {'surface', 'volumetric'}, optional
        Cortical surface source space, or a volume grid.
    spacing : str or float, optional
        Surface spacing (e.g. ``'oct6'``) or the grid step in mm.
    model : {'Single Layer', 'Triple Layer'}, optional
        BEM layers; EEG needs three.
    conductivity : tuple, optional
        Conductivities per layer, overriding ``model``.
    ico : int, optional
        BEM surface subdivision.
    mindist : float, optional
        Minimum distance of sources from the inner skull (mm).
    eeg : bool, optional
        Include EEG channels.
    """
    fs = FreeSurferFilenames(session)
    subjects_dir = str(fs.subjects_dir)
    if source_space in ("surface", "surf"):
        src = mne.setup_source_space(
            fs.subject,
            spacing=spacing,
            subjects_dir=subjects_dir,
            add_dist="patch",
            verbose=False,
        )
    elif source_space in ("volumetric", "vol"):
        src = mne.setup_volume_source_space(
            fs.subject, pos=float(spacing), subjects_dir=subjects_dir, verbose=False
        )
    else:
        raise ValueError("source_space must be 'surface' or 'volumetric'.")
    mne.write_source_spaces(
        str(fs.source_space_file), src, overwrite=True, verbose=False
    )

    if conductivity is None:
        conductivity = (0.3,) if model == "Single Layer" else (0.3, 0.006, 0.3)
    bem = mne.make_bem_solution(
        mne.make_bem_model(
            fs.subject,
            ico=ico,
            conductivity=conductivity,
            subjects_dir=subjects_dir,
            verbose=False,
        ),
        verbose=False,
    )
    info = mne.io.read_info(str(fs.info_file), verbose=False)
    fwd = mne.make_forward_solution(
        info,
        trans=str(fs.trans_file),
        src=src,
        bem=bem,
        meg=True,
        eeg=eeg,
        mindist=mindist,
        verbose=False,
    )
    mne.write_forward_solution(
        str(fs.fwd_model_file), fwd, overwrite=True, verbose=False
    )
    save_params(
        str(fs.fwd_model_params_file),
        {
            "source_space": source_space,
            "spacing": spacing,
            "model": model,
            "conductivity": list(conductivity),
            "ico": ico,
            "mindist": mindist,
            "eeg": eeg,
        },
    )


def minimum_norm(
    session: Session,
    chantypes: str | list[str],
    rank: dict | str,
    depth: float = 0.8,
    loose: float | str = "auto",
    frequency_range: list | None = None,
    data=None,
    fwd_file: str | None = None,
) -> mne.minimum_norm.InverseOperator:
    """Minimum-norm inverse operator from the forward model and the data's noise covariance.

    Parameters
    ----------
    session : Session
        Files of the session.
    chantypes : str or list of str
        Channel types to use.
    rank : dict or str
        Rank of the noise covariance, as for MNE.
    depth, loose : optional
        Depth weighting and loose orientation constraint.
    frequency_range : list, optional
        Band-pass applied to the data first.
    data : mne.io.Raw or mne.Epochs, optional
        The data; ``session.preproc_file`` by default.
    fwd_file : str, optional
        Forward model; the FreeSurfer-route one by default, or pass
        ``session.fwd_model_file`` for the RHINO one.
    """
    fs = FreeSurferFilenames(session)
    fs.root.mkdir(parents=True, exist_ok=True)
    chantypes = [chantypes] if isinstance(chantypes, str) else list(chantypes)
    data = _read_data(session, data, frequency_range)
    fwd = mne.read_forward_solution(fwd_file or str(fs.fwd_model_file), verbose=False)
    noise_cov = calc_noise_cov(data, rank, chantypes)
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        data.info, fwd, noise_cov, loose=loose, depth=depth, rank=rank, verbose=False
    )
    mne.minimum_norm.write_inverse_operator(
        str(fs.inverse_operator_file), inverse_operator, overwrite=True, verbose=False
    )
    save_params(
        str(fs.inverse_params_file),
        {
            "chantypes": chantypes,
            "rank": rank,
            "depth": depth,
            "loose": loose,
            "frequency_range": frequency_range,
            "fwd_file": fwd_file,
        },
    )
    return inverse_operator


_METHODS = {"mne": "MNE", "dspm": "dSPM", "sloreta": "sLORETA", "eloreta": "eLORETA"}


def apply_inverse(
    session: Session,
    data=None,
    method: str = "MNE",
    lambda2: float = 1.0 / 9,
    pick_ori: str | None = None,
    morph: str | None = "fsaverage",
    frequency_range: list | None = None,
):
    """Source estimate on the cortical surface, morphed to a subject (fsaverage by default).

    Returns
    -------
    stc : mne.SourceEstimate or list of them
        One estimate for raw data, one per epoch for epochs.
    src : mne.SourceSpaces
        The source space the estimate is on (the morph target's if morphed).
    """
    fs = FreeSurferFilenames(session)
    data = _read_data(session, data, frequency_range)
    inverse_operator = mne.minimum_norm.read_inverse_operator(
        str(fs.inverse_operator_file), verbose=False
    )
    method = _METHODS.get(str(method).lower(), method)
    apply = (
        mne.minimum_norm.apply_inverse_epochs
        if isinstance(data, mne.BaseEpochs)
        else mne.minimum_norm.apply_inverse_raw
    )
    stc = apply(
        data,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        pick_ori=pick_ori,
        verbose=False,
    )
    src = mne.read_source_spaces(str(fs.source_space_file), verbose=False)
    if morph and morph != fs.subject:
        spacing = (
            src[0]["info"]
            .get("command_line", "oct6")
            .split("spacing=")[-1]
            .split(",")[0]
            .strip("'\" )")
            if morph == "fsaverage"
            else None
        )
        src_to = (
            fsaverage_source_space(session, spacing or "oct6")
            if morph == "fsaverage"
            else None
        )
        m = mne.compute_source_morph(
            src,
            subject_from=fs.subject,
            subject_to=morph,
            subjects_dir=str(fs.subjects_dir),
            src_to=src_to,
            verbose=False,
        )
        m.save(str(fs.morph_file), overwrite=True, verbose=False)
        stc = [m.apply(s) for s in stc] if isinstance(stc, list) else m.apply(stc)
        src = src_to if src_to is not None else src
    return stc, src


def apply_inverse_vol(
    session: Session,
    data=None,
    method: str = "MNE",
    lambda2: float = 1.0 / 9,
    frequency_range: list | None = None,
):
    """Volumetric minimum-norm estimate, one time course per dipole along its
    maximum-variance orientation.

    Returns
    -------
    ts : (n_dipoles, n_times) numpy.ndarray
        Source time courses (bad annotations omitted).
    coords : (n_dipoles, 3) numpy.ndarray
        Dipole coordinates (mm) in the source space's frame.
    """
    fs = FreeSurferFilenames(session)
    data = _read_data(session, data, frequency_range)
    if isinstance(data, mne.BaseEpochs):
        raise ValueError("Volumetric minimum norm supports raw data only.")
    inverse_operator = mne.minimum_norm.read_inverse_operator(
        str(fs.inverse_operator_file), verbose=False
    )
    method = _METHODS.get(str(method).lower(), method)
    stc = mne.minimum_norm.apply_inverse_raw(
        data,
        inverse_operator,
        lambda2=lambda2,
        pick_ori="vector",
        method=method,
        method_params={"eps": 1e-8, "max_iter": 100},
        verbose=False,
    )
    xyz = stc.data - stc.data.mean(axis=-1, keepdims=True)
    source_nn = inverse_operator["source_nn"][2::3]
    eig_vals, eig_vecs = np.linalg.eig(np.matmul(xyz, xyz.transpose(0, 2, 1)))
    order = np.argsort(np.abs(eig_vals), axis=-1)
    ori = eig_vecs[np.arange(len(eig_vecs)), :, order[:, -1]].real
    signs = np.sign(np.sum(ori * source_nn, axis=1, keepdims=True))
    signs[signs == 0] = 1.0
    ori *= signs
    ts = np.squeeze(np.matmul(np.expand_dims(ori, 1), xyz), axis=1)
    _, times = data.get_data(
        reject_by_annotation="omit", return_times=True, verbose=False
    )
    ts = ts[:, np.isin(stc.times, times)]
    return ts


def parcellate(
    session: Session,
    parcellation_file: str = "aparc",
    method: str = "pca_flip",
    morph: str = "fsaverage",
    inverse_method: str = "MNE",
    lambda2: float = 1.0 / 9,
    pick_ori: str | None = None,
    data=None,
    frequency_range: list | None = None,
):
    """Parcel time courses from a surface minimum-norm estimate.

    Parameters
    ----------
    parcellation_file : str, optional
        A FreeSurfer annotation of the morph target (e.g. ``aparc``).
    method : str, optional
        How a label's vertices are combined: ``pca_flip``, ``mean``,
        ``mean_flip``, ``max`` or ``auto``.
    morph : str, optional
        Subject whose annotation is used, ``fsaverage`` by default.

    Returns
    -------
    parcel_data : (n_parcels, n_times) numpy.ndarray
    labels : list of str
        Parcel names.
    data : the preprocessed data the estimate came from
    """
    fs = FreeSurferFilenames(session)
    data = _read_data(session, data, frequency_range)
    stc, src = apply_inverse(session, data, inverse_method, lambda2, pick_ori, morph)
    subject = morph or fs.subject
    labels = mne.read_labels_from_annot(
        subject,
        parc=parcellation_file,
        subjects_dir=str(fs.subjects_dir),
        verbose=False,
    )
    labels = [l for l in labels if not l.name.startswith("unknown")]
    parcel_data = mne.extract_label_time_course(
        stc, labels, src, mode=method, verbose=False
    )
    if isinstance(parcel_data, list):
        parcel_data = np.array(parcel_data)
    return parcel_data, [l.name for l in labels], data


def calc_noise_cov(data, data_cov_rank, chantypes):
    """Calculate noise covariance.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw object.
    data_cov_rank : int
        Rank of the data covariance matrix.
    chantypes : list
        List of channel types to include.
    """
    # In MNE, the noise cov is normally obtained from empty room noise
    # recordings or from a baseline period. Here (if no noise cov is passed in)
    # we mimic what the osl_normalise_sensor_data.m function in Matlab OSL does,
    # by computing a diagonal noise cov with the variances set to the mean
    # variance of each sensor type (e.g. mag, grad, eeg.)
    logger.info("*** Calculating noise covariance ***")

    data = data.pick(chantypes)
    if isinstance(data, mne.io.Raw):
        data_cov = mne.compute_raw_covariance(
            data, method="empirical", rank=data_cov_rank
        )
    else:
        data_cov = mne.compute_covariance(data, method="empirical", rank=data_cov_rank)

    n_channels = data_cov.data.shape[0]
    noise_cov_diag = np.zeros(n_channels)

    for type in chantypes:
        # Indices of this channel type
        type_raw = data.copy().pick(type, exclude="bads")
        inds = []
        for chan in type_raw.info["ch_names"]:
            inds.append(data_cov.ch_names.index(chan))

        # Mean variance of channels of this type
        variance = np.mean(np.diag(data_cov.data)[inds])
        noise_cov_diag[inds] = variance
        logger.info(f"variance for chantype {type} is {variance}")

    bads = [b for b in data.info["bads"] if b in data_cov.ch_names]

    noise_cov = mne.Covariance(
        noise_cov_diag,
        data_cov.ch_names,
        bads,
        data.info["projs"],
        nfree=data.n_times,
    )

    return noise_cov


def is_freesurfer_route(dataset: dict, userargs: dict) -> bool:
    """Whether a step takes the FreeSurfer route: the step's own
    ``surface_extraction_method`` option, else the run's."""
    method = userargs.pop("surface_extraction_method", None) or dataset.get(
        "surface_extraction_method", "fsl"
    )
    return method == "freesurfer"
