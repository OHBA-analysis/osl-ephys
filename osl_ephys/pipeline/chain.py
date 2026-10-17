"""Run a config on one session."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path

import mne

from osl_ephys.pipeline import config as cfg
from osl_ephys.pipeline import manifest, registry
from osl_ephys.utils import logger as osl_logger
from osl_ephys.io.file_handling import find_run_id
from osl_ephys.utils.misc import set_random_seed

logger = logging.getLogger("osl_ephys")

PREPROC_FTYPE = "preproc-raw"


def run_chain(
    config,
    session,
    outdir=None,
    qc_dir=None,
    log_dir=None,
    extra_funcs=None,
    covs=None,
    overwrite=False,
    random_seed="auto",
    verbose="INFO",
    mneverbose="WARNING",
    surface_extraction_method="fsl",
):
    """Run the per-session sections of a config on one session.

    Parameters
    ----------
    config : str or dict
        The config (see :mod:`osl_ephys.pipeline.config`).
    session : dict, str or mne.io.Raw
        A session from the manifest, the path of a raw file, or a Raw
        object already in memory.
    outdir : str, optional
        Output directory. Without it nothing is written: the preprocessing
        section runs in memory and the sections after it cannot run.
    qc_dir : str, optional
        Directory for the QC report; no report is made without it.
    log_dir : str, optional
        Directory for the log files of the source-reconstruction steps.
    extra_funcs : list of callable, optional
        Custom steps.
    covs : dict or pandas.DataFrame, optional
        Covariates for the GLM steps.
    overwrite : bool, optional
        Rerun the preprocessing when its output exists.
    random_seed : 'auto', int or None, optional
        Seed for Python and NumPy; 'auto' draws one.
    verbose, mneverbose : str, optional
        Logging levels of osl-ephys and MNE.
    surface_extraction_method : {'fsl', 'freesurfer'}, optional
        Passed to the source-reconstruction steps.

    Returns
    -------
    dataset : dict
        What the preprocessing produced (``raw``, ``events``, ``epochs``,
        ``event_id``, ``ica``, ``fig``) plus ``session`` and ``paths``.
    """
    config = cfg.load_config(config)
    cfg.validate(config, extra_funcs)
    session = _as_session(session)
    paths = manifest.output_paths(session, outdir) if outdir is not None else {}

    # Inside a worker the session logger already captures the package logger;
    # setting it up again would point its handlers at the previous session
    if getattr(osl_logger.osl_logger, "already_setup", False):
        osl_logger.set_level(verbose)
    else:
        osl_logger.set_up(prefix=session["id"], level=verbose, startup=False)
    mne.set_log_level(mneverbose)
    if random_seed == "auto":
        set_random_seed()
    elif random_seed is not None:
        set_random_seed(random_seed)

    dataset = {
        "raw": session.pop("raw", None),
        "events": None,
        "epochs": None,
        "event_id": config["meta"]["event_codes"],
        "ica": None,
        "fig": {},
        "covs": covs,
        "session": session,
        "paths": paths,
        "qc_dir": qc_dir,
        "log_dir": log_dir,
        "surface_extraction_method": surface_extraction_method,
    }

    if cfg.steps(config, "maxfilter"):
        if outdir is None:
            raise ValueError(
                "The maxfilter section writes to outdir, which was not given."
            )
        for name, userargs in cfg.steps(config, "maxfilter"):
            step = registry.find_function(name, "maxfilter", extra_funcs)
            logger.info(f"Running {name}")
            dataset = step.func(dataset, deepcopy(userargs))

    if cfg.steps(config, "preproc"):
        dataset = _run_preproc(config, dataset, qc_dir, extra_funcs, overwrite)

    later = [
        s for s in ("coreg", "source_recon", "parcellation") if cfg.steps(config, s)
    ]
    if later:
        if outdir is None:
            raise ValueError(
                f"Sections {later} write their output to outdir, which was not given."
            )
        _run_source_recon(
            config,
            later,
            dataset,
            outdir,
            qc_dir,
            log_dir,
            extra_funcs,
            surface_extraction_method,
        )

    return dataset


def _as_session(session) -> dict:
    """A manifest row from whatever the caller passed."""
    if isinstance(session, (str, Path)):
        return {"id": find_run_id(str(session)), "raw_file": str(session)}
    if isinstance(session, mne.io.BaseRaw):
        name = (
            session.filenames[0]
            if session.filenames and session.filenames[0]
            else "raw"
        )
        return {"id": find_run_id(str(name)), "raw": session}
    if "id" not in session:
        raise ValueError(f"The session has no 'id': {session}")
    return dict(session)


def _run_preproc(config, dataset, qc_dir, extra_funcs, overwrite):
    """The preprocessing section: load, run the steps, write, report."""
    from osl_ephys.preproc.dataset import (
        append_preproc_info,
        import_data,
        write_dataset,
    )

    session, paths = dataset["session"], dataset["paths"]
    out_file = paths.get("preproc_file")
    if out_file is not None and out_file.exists() and not overwrite:
        logger.info(
            f"Skipping preprocessing: {out_file} exists (pass overwrite=True to rerun)."
        )
        return dataset

    if dataset["raw"] is None:
        # The maxfiltered file when the maxfilter section ran, else the raw file
        raw_file = dataset.get("maxfilter_file") or session.get("raw_file")
        if raw_file is None:
            raise ValueError(
                f"Session '{session['id']}' has no raw_file and no raw object."
            )
        dataset["raw"] = import_data(str(raw_file))

    for name, userargs in cfg.steps(config, "preproc"):
        userargs = deepcopy(userargs)
        step = registry.find_function(
            name, "preproc", extra_funcs, target=userargs.get("target", "raw")
        )
        logger.info(f"Running {name}")
        dataset = step.func(dataset, userargs)

    dataset = append_preproc_info(dataset, config, extra_funcs)

    if out_file is not None:
        paths["preproc_dir"].mkdir(parents=True, exist_ok=True)
        outbase = str(paths["preproc_dir"] / "{run_id}_{ftype}.{fext}")
        outnames = write_dataset(
            dataset, outbase, session["id"], ftype=PREPROC_FTYPE, overwrite=overwrite
        )
        dataset["outnames"] = outnames
        if qc_dir is not None:
            _preproc_report(dataset, qc_dir, outnames)
    return dataset


def _preproc_report(dataset, qc_dir, outnames):
    """The per-session preprocessing QC: the numbers and plots for the report."""
    from osl_ephys.preproc.qc import save_qc_plots

    save_qc_plots(
        qc_dir,
        dataset["session"]["id"],
        dataset["raw"],
        ica=dataset["ica"],
        events=dataset["events"],
        event_id=dataset["event_id"],
        epochs=dataset["epochs"],
        custom_figures=dataset["fig"] or None,
    )


def _run_source_recon(
    config,
    sections,
    dataset,
    outdir,
    qc_dir,
    log_dir,
    extra_funcs,
    surface_extraction_method,
):
    """The coregistration, source reconstruction and parcellation sections."""
    for section in sections:
        for name, userargs in cfg.steps(config, section):
            step = registry.find_function(name, section, extra_funcs)
            logger.info(f"Running {name}")
            dataset = step.func(dataset, deepcopy(userargs))
