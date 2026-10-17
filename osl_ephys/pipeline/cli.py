"""The ``osl-ephys`` command.

Subcommands: ``run`` (a config over a manifest), ``maxfilter`` (MaxFilter
a set of files, a one-section config), ``ica review`` and ``ica apply`` (the
hand-labelled ICA components), ``func`` (one function on one file),
``download-data`` (fetch every data file up front) and ``check`` (test the
installation).
"""

from __future__ import annotations

import argparse
import sys


def _run(args):
    from osl_ephys.pipeline import run_batch

    results = run_batch(
        args.config,
        args.sessions,
        args.outdir,
        n_workers=args.n_workers,
        log_dir=args.log_dir,
        qc_dir=False if args.no_report else args.qc_dir,
        overwrite=args.overwrite,
        verbose=args.verbose,
        mneverbose=args.mneverbose,
        surface_extraction_method=args.surface_extraction_method,
    )
    return 1 if any(status == "failed" for _, status, _, _ in results) else 0


def _download_data(args):
    from osl_ephys.files._fetch import cache_directory, fetch_all

    print(f"Downloading the osl-ephys data files to {cache_directory()}")
    path = fetch_all()
    print(f"Done. Set OSL_DATA to {path} to use this cache elsewhere.")
    return 0


def _check(args):
    """Import every subpackage, show versions, fetch a test file, run a chain."""
    from importlib.metadata import version

    import mne

    import osl_ephys
    import osl_ephys.files as files
    import osl_ephys.glm
    import osl_ephys.group
    import osl_ephys.maxfilter
    import osl_ephys.parcellation
    import osl_ephys.preproc
    import osl_ephys.report
    import osl_ephys.rhino
    import osl_ephys.source_recon
    from osl_ephys.pipeline import run_chain

    print(
        f"osl-ephys {osl_ephys.__version__}, mne {mne.__version__}, python {sys.version.split()[0]}"
    )
    for name in ("numpy", "scipy", "sails", "glmtools", "fslpy", "pyvista"):
        try:
            print(f"  {name} {version(name)}")
        except Exception:
            print(f"  {name} missing")
    print(f"data cache: {files._fetch.cache_directory()}")
    raw_file = files.test_data.elekta
    print(f"test recording: {raw_file}")
    dataset = run_chain(
        {"preproc": [{"resample": {"sfreq": 100}}]}, raw_file, verbose="WARNING"
    )
    print(
        f"preprocessing chain ran: {dataset['raw'].n_times} samples at {dataset['raw'].info['sfreq']} Hz"
    )
    print("osl-ephys is working.")
    return 0


def _func(args):
    """Call a function of the package from the command line, e.g.
    ``osl-ephys func osl_ephys.rhino.setup_fsl /opt/fsl``.
    Arguments are strings, except None, True and False."""
    module = __import__(args.function.split(".")[0])
    func = module
    for name in args.function.split(".")[1:]:
        func = getattr(func, name)
    literals = {"None": None, "True": True, "False": False}
    func(*[literals.get(a, a) for a in args.arguments])
    return 0


def _maxfilter(args):
    """MaxFilter the files, as the config ``maxfilter: [{maxfilter: options}]``."""
    from osl_ephys.pipeline import run_batch

    with open(args.files) as f:
        files = [line.strip() for line in f if line.strip()]
    options = {
        k: v
        for k, v in vars(args).items()
        if k not in ("files", "outdir", "n_workers", "func", "command")
        and v is not None
    }
    if "bads" in options:
        options["bads"] = " ".join(options["bads"])
    config = {"maxfilter": [{"maxfilter": options}]}
    results = run_batch(
        config, files, args.outdir, n_workers=args.n_workers, qc_dir=False
    )
    return 1 if any(status == "failed" for _, status, _, _ in results) else 0


def _ica_review(args):
    from osl_ephys.preproc.ica import review_gui, serve_review
    from osl_ephys.preproc.ica.files import sessions_with_ica

    if args.browser:
        serve_review(args.outdir, host=args.host, port=args.port)
        return 0
    ids = args.ids or sessions_with_ica(args.outdir)
    for id_ in ids:
        review_gui(
            args.outdir,
            id_,
            reject=None if args.reject == "none" else args.reject,
            run=args.run,
        )
    return 0


def _ica_apply(args):
    from osl_ephys.preproc.ica import apply_review
    from osl_ephys.preproc.ica.files import sessions_with_ica

    ids = args.ids or sessions_with_ica(args.outdir)
    failed = 0
    for id_ in ids:
        message = apply_review(args.outdir, id_, run=args.run, purge=args.purge)
        print(f"{id_}: {message}")
        failed += not message.startswith("ok")
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="osl-ephys", description="OHBA Software Library for M/EEG analysis."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a config on a set of sessions")
    run.add_argument("config", help="YAML config file")
    run.add_argument(
        "sessions",
        help="CSV file with one row per session (columns: id, subject, raw_file, smri_file, ...)",
    )
    run.add_argument("outdir", help="output directory")
    run.add_argument(
        "--n-workers", type=int, default=1, help="sessions processed at the same time"
    )
    run.add_argument(
        "--log-dir",
        default=None,
        help="directory for the log files (default: outdir/logs)",
    )
    run.add_argument(
        "--qc-dir",
        default=None,
        help="directory for the QC report (default: outdir/qc)",
    )
    run.add_argument(
        "--no-report", action="store_true", help="do not generate the QC report"
    )
    run.add_argument(
        "--overwrite", action="store_true", help="rerun sessions whose output exists"
    )
    run.add_argument("--verbose", default="INFO", help="logging level of osl-ephys")
    run.add_argument("--mneverbose", default="WARNING", help="logging level of MNE")
    run.add_argument(
        "--surface-extraction-method", default="fsl", choices=["fsl", "freesurfer"]
    )
    run.set_defaults(func=_run)

    mf = sub.add_parser("maxfilter", help="MaxFilter a set of fif files")
    mf.add_argument("files", help="text file with one raw fif file per line")
    mf.add_argument("outdir", help="output directory (files go to outdir/maxfilter)")
    mf.add_argument(
        "--n-workers", type=int, default=1, help="files processed at the same time"
    )
    mf.add_argument("--maxpath", help="the maxfilter command")
    mf.add_argument("--mode", choices=["standard", "multistage", "cbu"])
    mf.add_argument(
        "--scanner",
        choices=["VectorView", "VectorView2", "Neo"],
        help="OHBA scanner, sets ctc and cal",
    )
    mf.add_argument("--ctc", help="cross-talk calibration file")
    mf.add_argument("--cal", help="fine-calibration file")
    mf.add_argument("--tsss", action="store_true", default=None, help="temporal SSS")
    mf.add_argument("--st", type=float, help="tSSS buffer length (seconds)")
    mf.add_argument("--corr", type=float, help="tSSS subspace correlation limit")
    mf.add_argument(
        "--headpos",
        action="store_true",
        default=None,
        help="write the head position log",
    )
    mf.add_argument(
        "--movecomp", action="store_true", default=None, help="movement compensation"
    )
    mf.add_argument(
        "--movecompinter",
        action="store_true",
        default=None,
        help="movement compensation with intermittent HPI",
    )
    mf.add_argument(
        "--nomovecompinter",
        action="store_true",
        default=None,
        help="no movement compensation in the CBU recipe",
    )
    mf.add_argument(
        "--autobad",
        action="store_true",
        default=None,
        help="automatic bad channel detection",
    )
    mf.add_argument(
        "--autobad-dur", type=int, dest="autobad_dur", help="autobad duration (seconds)"
    )
    mf.add_argument("--bads", nargs="+", help="bad channel numbers")
    mf.add_argument(
        "--badlimit", type=float, help="autobad threshold (standard deviations)"
    )
    mf.add_argument(
        "--trans", help="transform to the head position of this file, or 'default'"
    )
    mf.add_argument("--origin", nargs=3, type=float, help="sphere origin x y z")
    mf.add_argument("--frame", help="coordinate frame of the origin")
    mf.add_argument(
        "--force", action="store_true", default=None, help="ignore MaxFilter warnings"
    )
    mf.add_argument("--inorder", type=int, help="order of the inside expansion")
    mf.add_argument("--outorder", type=int, help="order of the outside expansion")
    mf.add_argument("--hpie", type=int, help="HPI fit error limit (mm)")
    mf.add_argument("--hpig", type=float, help="HPI fit goodness-of-fit limit")
    mf.add_argument("--linefreq", type=int, help="line frequency (50 or 60)")
    mf.add_argument("--hpisubt", help="HPI signal subtraction mode")
    mf.add_argument("--skip", help="intervals to skip, 'start stop ...' in seconds")
    mf.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        help="overwrite existing output",
    )
    mf.add_argument(
        "--dryrun",
        action="store_true",
        default=None,
        help="print the commands without running them",
    )
    mf.set_defaults(func=_maxfilter)

    ica = sub.add_parser(
        "ica", help="review ICA components by hand and apply the labels"
    )
    ica_sub = ica.add_subparsers(dest="ica_command", required=True)
    review = ica_sub.add_parser(
        "review", help="label the components of a run's sessions"
    )
    review.add_argument("outdir", help="output directory of the run")
    review.add_argument(
        "ids", nargs="*", help="session ids (default: every session with an ICA)"
    )
    review.add_argument(
        "--reject",
        choices=["none", "all", "manual"],
        default="none",
        help="after the GUI: remove all labelled components, only those labelled now, or none",
    )
    review.add_argument(
        "--run", type=int, default=1, help="which ICA of a session with several"
    )
    review.add_argument(
        "--browser",
        action="store_true",
        help="serve the browser review pages of outdir/ica instead of the GUI",
    )
    review.add_argument("--host", default="127.0.0.1")
    review.add_argument("--port", type=int, default=8000)
    review.set_defaults(func=_ica_review)
    apply = ica_sub.add_parser(
        "apply", help="remove the labelled components from the data"
    )
    apply.add_argument("outdir", help="output directory of the run")
    apply.add_argument(
        "ids", nargs="*", help="session ids (default: every session with an ICA)"
    )
    apply.add_argument(
        "--run", type=int, default=1, help="which ICA of a session with several"
    )
    apply.add_argument(
        "--purge",
        action="store_true",
        help="delete the browser review's plots afterwards",
    )
    apply.set_defaults(func=_ica_apply)

    fn = sub.add_parser("func", help="call a function of the package")
    fn.add_argument("function", help="e.g. osl_ephys.rhino.setup_fsl")
    fn.add_argument("arguments", nargs="*", help="its arguments, as strings")
    fn.set_defaults(func=_func)

    sub.add_parser(
        "download-data", help="download every data file, for use offline"
    ).set_defaults(func=_download_data)
    sub.add_parser("check", help="check the installation works").set_defaults(
        func=_check
    )

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
