"""The parcellation steps of a config."""

from __future__ import annotations

import logging

import mne

from osl_ephys.parcellation import parcellation, plotting
from osl_ephys.source_recon import freesurfer
from osl_ephys.source_recon.beamforming import grid
from osl_ephys.source_recon.steps import beamform, minimum_norm
from osl_ephys.pipeline.registry import step
from osl_ephys.session import Session

logger = logging.getLogger("osl_ephys")


@step("parcellation")
def parcellate(dataset, userargs):
    """Parcel time courses from the source estimate, saved as ``<id>_parc-raw.fif``.

    Options: ``parcellation_file``, ``orthogonalisation`` (``symmetric`` or
    none), ``extra_chans`` (channels copied from the preprocessed data,
    ``stim`` by default), ``qc`` (save the QC plots, on by default), and
    ``source_method``: ``lcmv`` (default) uses the beamformer filters from
    the ``beamform`` step; ``minimum_norm`` applies the inverse operator
    from the ``minimum_norm`` step, volumetric on the RHINO route or on the
    surface on the FreeSurfer route, where ``parcellation_file`` is a
    FreeSurfer annotation (e.g. ``aparc``), ``method`` combines a label's
    vertices and ``morph`` is the subject whose annotation is used
    (fsaverage). Minimum-norm options: ``inverse_method``, ``lambda2``,
    ``pick_ori``.
    """
    session = Session.from_dataset(dataset)
    source_method = str(userargs.pop("source_method", "lcmv")).lower()
    extra_chans = userargs.pop("extra_chans", "stim")
    qc = userargs.pop("qc", True)
    orthogonalisation = userargs.pop("orthogonalisation", None)
    parcellation_file = userargs.pop("parcellation_file")
    fs_route = freesurfer.is_freesurfer_route(dataset, userargs)

    if source_method == "lcmv":
        for name in ("method", "morph", "inverse_method", "lambda2", "pick_ori"):
            userargs.pop(name, None)
        parcel_data = parcellation.parcellate_lcmv(
            session, parcellation_file, orthogonalisation=orthogonalisation, **userargs
        )
        names = None
        data = mne.io.read_raw_fif(session.preproc_file, preload=True, verbose=False)
    elif source_method in ("minimum_norm", "mne"):
        if fs_route:
            parcel_data, names, data = freesurfer.parcellate(
                session, parcellation_file, **userargs
            )
        else:
            userargs.pop("method", None)
            userargs.pop("morph", None)
            ts = freesurfer.apply_inverse_vol(
                session,
                method=userargs.pop("inverse_method", "MNE"),
                lambda2=userargs.pop("lambda2", 1.0 / 9),
            )
            # The dipoles are those of the RHINO forward model, on the MNI grid
            fwd = mne.read_forward_solution(session.fwd_model_file, verbose=False)
            coords = grid._get_source_coords_mni(session, fwd)
            parcel_data = parcellation.parcellate(
                ts, coords, parcellation_file, method="pca", orthogonalisation=None
            )
            names = None
            data = mne.io.read_raw_fif(
                session.preproc_file, preload=True, verbose=False
            )
        if orthogonalisation == "symmetric":
            parcel_data = parcellation._symmetric_orthogonalisation(parcel_data)
        elif orthogonalisation not in (None, "none", "None"):
            raise NotImplementedError(f"orthogonalisation '{orthogonalisation}'")
    else:
        raise ValueError("source_method must be 'lcmv' or 'minimum_norm'.")

    parc_file = dataset["paths"]["parc_file"]
    parc_file.parent.mkdir(parents=True, exist_ok=True)
    parc_raw = parcellation.convert_to_mne_raw(
        parcel_data, data, ch_names=names, extra_chans=extra_chans
    )
    parc_raw.save(str(parc_file), overwrite=True, verbose=False)
    if qc and dataset.get("qc_dir") is not None and not fs_route:
        # The dipoles are those of the RHINO forward model, on the MNI grid
        fwd = mne.read_forward_solution(session.fwd_model_file, verbose=False)
        counts = parcellation.parcel_dipole_counts(
            parcellation_file, grid._get_source_coords_mni(session, fwd)
        )
        plotting.save_qc_plots(
            dataset["qc_dir"],
            session.id,
            str(parc_file),
            parcellation_file,
            summary={
                "min_dipoles_per_parcel": int(counts.min()),
                "n_empty_parcels": int((counts == 0).sum()),
                "dipoles_per_parcel": counts.tolist(),
            },
        )
    return dataset


_BEAMFORM_OPTIONS = (
    "chantypes",
    "rank",
    "pick_ori",
    "reduce_rank",
    "frequency_range",
    "use_bilateral_pairs",
    "bilateral_tol",
    "bilateral_tol_midline",
    "weight_norm",
    "reg",
)


_MINIMUM_NORM_OPTIONS = (
    "chantypes",
    "rank",
    "depth",
    "loose",
    "frequency_range",
)


@step("parcellation")
def beamform_and_parcellate(dataset, userargs):
    """``beamform`` then ``parcellate``, with the options of both."""
    beam_args = {k: userargs.pop(k) for k in list(userargs) if k in _BEAMFORM_OPTIONS}
    dataset = beamform(dataset, beam_args)
    return parcellate(dataset, {**userargs, "source_method": "lcmv"})


@step("parcellation")
def minimum_norm_and_parcellate(dataset, userargs):
    """``minimum_norm`` then ``parcellate``, with the options of both."""
    mn_args = {k: userargs.pop(k) for k in list(userargs) if k in _MINIMUM_NORM_OPTIONS}
    if mn_args:
        dataset = minimum_norm(dataset, mn_args)
    return parcellate(dataset, {**userargs, "source_method": "minimum_norm"})
