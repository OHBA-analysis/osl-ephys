"""Removing the reviewed components from a session's preprocessed data."""

from __future__ import annotations

import logging
from copy import deepcopy

import mne

from osl_ephys.preproc.ica.files import ica_files, review_files, session_files

logger = logging.getLogger("osl_ephys")


def apply_review(outdir, id, exclude=None, run=1, purge=False):
    """Remove a session's labelled components from its preprocessed data.

    The components come from the browser review (``outdir/ica/<id>/label.txt``,
    with its bad segments annotated ``BAD_manual``) when it exists, else
    from the ICA file's ``exclude`` as saved by the GUI. The preprocessed
    file is overwritten and the session's QC ICA plot updated.

    Parameters
    ----------
    outdir : str or Path
        The output directory of the run.
    id : str
        The session id.
    exclude : list of int, optional
        The components to remove instead of the labelled ones.
    run : int, optional
        Which ICA of the session, for a session with several.
    purge : bool, optional
        Delete the browser review's plots afterwards (the labels are kept).

    Returns
    -------
    message : str
        What was done.
    """
    from osl_ephys.preproc.ica.manual.apply import apply_one
    from osl_ephys.preproc.qc import save_qc_plots

    preproc_file, qc_dir = session_files(outdir, id)
    ica_file = ica_files(outdir, id)[run - 1]
    review = review_files(outdir, id)
    if review is not None and exclude is None:
        message = apply_one(
            qc_dir.parent / "ica",
            None,
            id,
            overwrite=True,
            purge_svgs=purge,
            raw_path=preproc_file,
            ica_path=ica_file,
            out_path=preproc_file,
        )
        if not message.startswith("ok"):
            logger.warning(f"{id}: {message}")
            return message
        ica = mne.preprocessing.read_ica(ica_file, verbose=False)
        raw = mne.io.read_raw_fif(preproc_file, preload=True, verbose=False)
    else:
        ica = mne.preprocessing.read_ica(ica_file, verbose=False)
        if exclude is not None:
            ica.exclude = sorted(exclude)
        raw = mne.io.read_raw_fif(preproc_file, preload=True, verbose=False)
        ica.apply(raw, verbose=False)
        raw.save(preproc_file, overwrite=True, verbose=False)
        ica.save(ica_file, overwrite=True, verbose=False)
        message = f"ok: removed {len(ica.exclude)} components {sorted(ica.exclude)}"
    logger.info(f"{id}: {message}")
    if (qc_dir / "preproc" / id).is_dir():
        save_qc_plots(qc_dir, id, raw, ica=deepcopy(ica), plots=["ica"])
    return message
