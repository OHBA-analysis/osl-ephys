"""Run a config on every session of a manifest, in parallel."""

from __future__ import annotations

import logging
from copy import deepcopy
from functools import partial
from pathlib import Path

from osl_ephys.pipeline import chain
from osl_ephys.pipeline import config as cfg
from osl_ephys.pipeline import manifest, registry, runner
from osl_ephys.utils import logger as osl_logger

logger = logging.getLogger("osl_ephys")


def run_batch(
    config,
    sessions,
    outdir,
    n_workers=1,
    log_dir=None,
    qc_dir=None,
    extra_funcs=None,
    covs=None,
    overwrite=False,
    random_seed="auto",
    verbose="INFO",
    mneverbose="WARNING",
    surface_extraction_method="fsl",
):
    """Run a config on every session, then the group steps, then the report.

    Parameters
    ----------
    config : str or dict
        The config (see :mod:`osl_ephys.pipeline.config`).
    sessions : list of dict, str or list of str
        The manifest (see :mod:`osl_ephys.pipeline.manifest`).
    outdir : str
        Output directory.
    n_workers : int, optional
        Sessions processed at the same time.
    log_dir : str, optional
        Directory for the per-session log files and ``summary.csv``;
        ``outdir/logs`` by default.
    qc_dir : str, optional
        Directory for the QC report; ``outdir/qc`` by default. Pass
        ``False`` for no report.
    extra_funcs : list of callable, optional
        Custom steps.
    covs : dict or pandas.DataFrame, optional
        Covariates for the GLM steps.
    overwrite : bool, optional
        Rerun sessions whose preprocessing output exists.
    random_seed : 'auto', int or None, optional
        Seed for Python and NumPy in every worker; 'auto' draws one.
    verbose, mneverbose : str, optional
        Logging levels of osl-ephys and MNE.
    surface_extraction_method : {'fsl', 'freesurfer'}, optional
        Passed to the source-reconstruction steps.

    Returns
    -------
    results : list of tuple
        For each session: its id, status ("ok", "partial" or "failed"),
        time taken in seconds, and the warnings or error message.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(log_dir) if log_dir is not None else outdir / "logs"
    qc_dir = (
        None
        if qc_dir is False
        else Path(qc_dir) if qc_dir is not None else outdir / "qc"
    )

    config = cfg.load_config(config)
    sessions = manifest.load_manifest(sessions)
    cfg.validate(config, extra_funcs, manifest.columns(sessions))
    cfg.check_versions(config)

    osl_logger.set_up(level=verbose, startup=False)
    logger.info(
        f"Running {len(sessions)} sessions with {n_workers} workers; logs in {log_dir}"
    )

    if random_seed == "auto":
        from osl_ephys.utils.misc import set_random_seed

        random_seed = set_random_seed()

    common = dict(
        outdir=outdir,
        qc_dir=qc_dir,
        log_dir=log_dir,
        extra_funcs=extra_funcs,
        covs=covs,
        overwrite=overwrite,
        random_seed=random_seed,
        verbose=verbose,
        mneverbose=mneverbose,
        surface_extraction_method=surface_extraction_method,
    )

    # Steps that run once per subject go first, on the first session of each
    # subject; the sessions then share their output
    subject_config, session_config = _split_by_scope(config, extra_funcs)
    if any(cfg.steps(subject_config, s) for s in cfg.SECTION_ORDER):
        firsts = [rows[0] for rows in manifest.by_subject(sessions).values()]
        runner.run(
            partial(_process, config=subject_config, **common),
            firsts,
            n_workers,
            log_dir / "subjects",
        )

    results = runner.run(
        partial(_process, config=session_config, **common), sessions, n_workers, log_dir
    )

    if cfg.steps(config, "group"):
        ok = {id_ for id_, status, _, _ in results if status != "failed"}
        _run_group(
            config,
            [s for s in sessions if s["id"] in ok],
            outdir,
            extra_funcs,
            covs,
            overwrite,
            qc_dir,
        )

    if qc_dir is not None:
        _report(qc_dir, sessions, outdir)
    return results


def _process(session, config, **kwargs):
    """One session in a worker: everything it prints goes to its log file."""
    chain.run_chain(config, session, **kwargs)


def _split_by_scope(config, extra_funcs):
    """The config split into its per-subject steps and its per-session steps."""
    subject_config = {"meta": config["meta"]}
    session_config = {"meta": config["meta"]}
    for section in cfg.SECTION_ORDER:
        if section == "group":
            continue
        for name, kwargs in cfg.steps(config, section):
            step = registry.find_function(
                name, section, extra_funcs, target=kwargs.get("target", "raw")
            )
            target = subject_config if step.per == "subject" else session_config
            target.setdefault(section, []).append({name: kwargs})
    return subject_config, session_config


def _run_group(config, sessions, outdir, extra_funcs, covs, overwrite, qc_dir):
    """The group section, on the outputs of the sessions that finished."""
    from osl_ephys.preproc.dataset import write_dataset

    logger.info(f"Running the group steps on {len(sessions)} sessions")
    paths = [manifest.output_paths(s, outdir) for s in sessions]
    # The sessions' fitted GLMs, for the group GLM steps
    glm_files = [
        p["preproc_dir"] / f"{s['id']}_glm.pkl" for s, p in zip(sessions, paths)
    ]
    dataset = {
        "raw": [str(p["preproc_file"]) for p in paths],
        "glm": (
            [str(f) for f in glm_files] if all(f.exists() for f in glm_files) else None
        ),
        "covs": covs,
        "fig": {},
        "sessions": sessions,
        "outdir": outdir,
        "qc_dir": qc_dir,
    }
    inputs = set(dataset)
    for name, userargs in cfg.steps(config, "group"):
        userargs = {**deepcopy(userargs), "run_on_group": True}
        step = registry.find_function(
            name, "group", extra_funcs, target=userargs.get("target", "raw")
        )
        dataset = step.func(dataset, userargs)
    # Only what the group steps produced is written, next to the sessions
    produced = [k for k in dataset if k not in inputs and k not in ("fig", "template")]
    if produced:
        outbase = str(outdir / "{ftype}.{fext}")
        write_dataset(
            dataset,
            outbase,
            "",
            ftype="",
            overwrite=overwrite,
            skip=[k for k in dataset if k not in produced],
        )


def _report(qc_dir, sessions, outdir):
    """The QC report across sessions, from what the steps saved."""
    from osl_ephys.report import generate_report

    generate_report(qc_dir, sessions, outdir)
