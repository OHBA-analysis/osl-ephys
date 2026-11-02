"""The coreg steps of a config and the RHINO forward model (a source_recon step).

Each step takes the dataset dict of a session and its options, builds the
:class:`~osl_ephys.session.Session` that holds the session's files and calls
the function that does the work. Steps that have a FreeSurfer route dispatch
on the run's ``surface_extraction_method``."""

from __future__ import annotations

import logging

import numpy as np

from osl_ephys import rhino
from osl_ephys.source_recon import freesurfer
from osl_ephys.pipeline.registry import step
from osl_ephys.session import Session

logger = logging.getLogger("osl_ephys")


@step("coreg", per="subject")
def extract_surfaces(dataset, userargs):
    """Extract the head surfaces from the subject's structural MRI with FSL.

    Options: ``include_nose``, ``do_mri2mniaxes_xform``, ``bet_fval``,
    ``nonlinear_registration``. Skipped for a session with no ``smri_file``,
    which uses the MNI152 surfaces.
    """
    row = dataset["session"]
    if not row.get("smri_file"):
        logger.info(f"{row['id']} has no structural MRI; the MNI152 surfaces are used.")
        return dataset
    rhino.extract_surfaces(
        str(row["smri_file"]), str(dataset["paths"]["surfaces_dir"]), **userargs
    )
    return dataset


@step("coreg")
def extract_fiducials_and_headshape(dataset, userargs):
    """Fiducials and headshape points from the preprocessed file's digitisation.

    Options: ``include_eeg_as_headshape``, ``include_hpi_as_headshape``,
    ``include_extra_as_headshape``.
    """
    rhino.extract_fiducials_and_headshape_from_fif(
        Session.from_dataset(dataset), **userargs
    )
    return dataset


@step("coreg")
def extract_fiducials_and_headshape_from_pos(dataset, userargs):
    """Fiducials and headshape points from the session's ``pos_file``."""
    rhino.extract_fiducials_and_headshape_from_pos(Session.from_dataset(dataset))
    return dataset


@step("coreg")
def extract_fiducials_and_headshape_from_elc(dataset, userargs):
    """Fiducials and headshape points from the session's ``elc_file``. Option: ``remove_nose``."""
    rhino.extract_fiducials_and_headshape_from_elc(
        Session.from_dataset(dataset), **userargs
    )
    return dataset


@step("coreg")
def remove_stray_headshape_points(dataset, userargs):
    """Remove headshape points on the nose, neck or far from the head. Options: ``nose``, ``neck``, ``far``."""
    rhino.remove_stray_headshape_points(Session.from_dataset(dataset), **userargs)
    return dataset


@step("coreg")
def save_mni_fiducials(dataset, userargs):
    """Use fiducials given in MNI coordinates for the coregistration.

    Option: ``filepath``, a text file with the nasion, LPA and RPA as rows
    (mm, MNI space). The coregistration step maps them to the subject.
    """
    coords = np.loadtxt(userargs["filepath"])
    dataset["mni_fiducials"] = {"nasion": coords[0], "lpa": coords[1], "rpa": coords[2]}
    return dataset


@step("coreg")
def coregister(dataset, userargs):
    """Coregister the head (sensors and headshape) with the MRI.

    RHINO route options: ``use_headshape``, ``use_nose``,
    ``allow_mri_scaling``, ``n_init``,
    ``plot_type``, ``already_coregistered`` (just save the files RHINO
    needs). FreeSurfer route options: ``allow_mri_scaling``, ``n_init``,
    ``fit_fiducials``, ``fit_icp``.
    """
    session = Session.from_dataset(dataset)
    if freesurfer.is_freesurfer_route(dataset, userargs):
        for name in ("use_headshape", "use_nose", "plot_type", "already_coregistered"):
            userargs.pop(name, None)
        if userargs.get("n_init") is None:
            userargs["n_init"] = 20
        freesurfer.coregister(session, **userargs)
        return dataset
    if userargs.pop("already_coregistered", False):
        rhino.save_coregistration_files(session)
        return dataset
    if userargs.get("n_init") is None:
        userargs["n_init"] = 1
    rhino.coregister_head_and_mri(
        session, mni_fiducials=dataset.get("mni_fiducials"), **userargs
    )
    return dataset


@step("source_recon")
def forward_model(dataset, userargs):
    """Forward model.

    RHINO route (volumetric grid): ``model``, ``gridstep``, ``mindist``,
    ``exclude``, ``eeg``, ``meg``. FreeSurfer route, or
    ``source_space: surface``: ``source_space``, ``spacing``, ``model``, ``conductivity``, ``ico``, ``mindist``,
    ``eeg``.
    """
    if freesurfer.is_freesurfer_route(dataset, userargs) or userargs.get(
        "source_space"
    ) in (
        "surface",
        "surf",
    ):
        userargs.pop("meg", None)
        userargs.pop("exclude", None)
        freesurfer.forward_model(Session.from_dataset(dataset), **userargs)
        return dataset
    userargs.pop("source_space", None)
    rhino.forward_model(Session.from_dataset(dataset), **userargs)
    return dataset
