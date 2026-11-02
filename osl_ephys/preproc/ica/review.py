"""Reviewing a session's ICA: the GUI, or the browser pages' server."""

from __future__ import annotations

import logging
from pathlib import Path

import mne

from osl_ephys.preproc.ica.files import ica_files, session_files

logger = logging.getLogger("osl_ephys")


def review_gui(outdir, id, reject=None, run=1, stop=30):
    """Label the components of a session's ICA in the GUI.

    Opens :func:`~osl_ephys.preproc.ica.plot_ica` on the session's ICA and
    preprocessed data (blocking until the window is closed), saves the ICA
    with its labels, and optionally removes the labelled components from
    the data.

    Parameters
    ----------
    outdir : str or Path
        The output directory of the run.
    id : str
        The session id.
    reject : {None, "all", "manual"}, optional
        None saves the labels only; ``all`` removes every labelled
        component from the data; ``manual`` removes only the components
        labelled in this review (not those already excluded).
    run : int, optional
        Which ICA of the session to review, for a session with several.
    stop : float, optional
        Seconds of data shown at first.
    """
    import matplotlib.pyplot as plt

    from osl_ephys.preproc.ica.gui import plot_ica

    preproc_file, _ = session_files(outdir, id)
    ica_file = ica_files(outdir, id)[run - 1]
    logger.info(f"Reviewing {ica_file}")
    raw = mne.io.read_raw_fif(preproc_file, preload=True, verbose=False)
    ica = mne.preprocessing.read_ica(ica_file, verbose=False)
    already = list(ica.exclude)

    plt.ion()
    plot_ica(ica, raw, block=True, stop=stop)
    plt.pause(0.1)

    # labels_ must hold lists for the ICA to be saved
    for key in ica.labels_:
        ica.labels_[key] = list(ica.labels_[key])
    ica.save(ica_file, overwrite=True, verbose=False)
    logger.info(f"Saved the labels: {sorted(ica.exclude)} excluded")

    if reject in ("all", "manual"):
        from osl_ephys.preproc.ica.apply import apply_review

        exclude = (
            sorted(set(ica.exclude) - set(already)) if reject == "manual" else None
        )
        apply_review(outdir, id, exclude=exclude, run=run)


def serve_review(outdir, host="127.0.0.1", port=8000):
    """Serve the browser review pages of a run (``outdir/ica``), saving the
    labels until interrupted.

    Parameters
    ----------
    outdir : str or Path
        The output directory of the run; the pages are under ``outdir/ica``.
    host : str, optional
        The interface to bind; ``0.0.0.0`` exposes the pages on the network.
    port : int, optional
        The port.
    """
    from osl_ephys.preproc.ica.manual.review_server import create_server

    root = Path(outdir) / "ica"
    if not root.is_dir():
        raise FileNotFoundError(
            f"No review pages under {root}; run the manual_ica step first."
        )
    server = create_server(host, port, root)
    print(f"ICA review: http://{host}:{port}/<id>/single_ic.html (Ctrl-C to stop)")
    if host == "0.0.0.0":
        print("Bound to 0.0.0.0: the save endpoints are reachable from the network.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
