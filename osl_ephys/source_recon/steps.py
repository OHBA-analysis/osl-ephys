"""The source_recon steps of a config: the beamformer and minimum norm, and the
FreeSurfer surface steps (coreg)."""

from __future__ import annotations

import logging

from osl_ephys.source_recon import beamforming, freesurfer
from osl_ephys.pipeline.registry import step
from osl_ephys.session import Session

logger = logging.getLogger("osl_ephys")


@step("coreg", per="subject")
def recon_all(dataset, userargs):
    """FreeSurfer's ``recon-all`` on the subject's structural MRI (needs FreeSurfer).

    A session with no ``smri_file`` gets MNE's fsaverage instead.
    """
    row = dataset["session"]
    if not row.get("smri_file"):
        logger.info(f"{row['id']} has no structural MRI; fsaverage is used.")
        freesurfer.fetch_fsaverage(Session.from_dataset(dataset))
        return dataset
    freesurfer.recon_all(Session.from_dataset(dataset), str(row["smri_file"]))
    return dataset


@step("coreg", per="subject")
def make_watershed_bem(dataset, userargs):
    """FreeSurfer's watershed BEM surfaces for the subject (needs FreeSurfer)."""
    freesurfer.make_watershed_bem(Session.from_dataset(dataset), **userargs)
    return dataset


@step("source_recon")
def minimum_norm(dataset, userargs):
    """Minimum-norm inverse operator from the forward model.

    Options: ``chantypes``, ``rank``, ``depth``, ``loose``,
    ``frequency_range``. The RHINO route uses the
    volumetric forward model, the FreeSurfer route the surface one.
    ``lambda2`` and ``pick_ori`` belong to the parcellation step.
    """
    session = Session.from_dataset(dataset)
    fwd_file = (
        None
        if freesurfer.is_freesurfer_route(dataset, userargs)
        else session.fwd_model_file
    )
    freesurfer.minimum_norm(session, fwd_file=fwd_file, **userargs)
    return dataset


@step("source_recon")
def beamform(dataset, userargs):
    """LCMV beamformer filters, saved with the session.

    Options: ``chantypes``, ``rank``, ``pick_ori``, ``reduce_rank``,
    ``frequency_range``, ``use_bilateral_pairs``,
    ``bilateral_tol``, ``bilateral_tol_midline``.
    """
    beamforming.lcmv_beamformer(Session.from_dataset(dataset), **userargs)
    return dataset
