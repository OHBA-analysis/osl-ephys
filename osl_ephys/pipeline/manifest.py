"""The session manifest: which files each session has and where its outputs go.

A manifest is a list of dicts, one per session, with these columns:

- ``id`` (required): the session name, used for output files and logs.
- ``subject``: the subject a session belongs to, for steps that run once per
  subject (surface extraction) and for the head model shared by a subject's
  sessions.
- ``raw_file``: the raw recording, for ``preproc``.
- ``smri_file``: the structural MRI, for surface extraction.
- ``preproc_file``: the preprocessed file, when it is not at the standard
  location under ``outdir``.
- ``epoch_file``, ``pos_file``, ``elc_file``, ``localizer_file``: optional
  inputs of the source-reconstruction steps.

Any other column is passed through to the steps in the dataset.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

from osl_ephys.io.file_handling import find_run_id

COLUMNS = (
    "id",
    "subject",
    "raw_file",
    "smri_file",
    "preproc_file",
    "epoch_file",
    "pos_file",
    "elc_file",
    "localizer_file",
)


def load_manifest(sessions) -> list[dict]:
    """A manifest from a list of dicts, a CSV file, or a list of raw files.

    Parameters
    ----------
    sessions : list of dict, str, or list of str
        The sessions. A string is the path of a CSV file with a header row;
        a list of strings is a list of raw files whose ids are derived from
        the file names.

    Returns
    -------
    sessions : list of dict
        One dict per session with at least ``id``.
    """
    if isinstance(sessions, (str, Path)):
        with open(sessions, newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
    elif all(isinstance(s, (str, Path)) for s in sessions):
        rows = [{"id": find_run_id(str(s)), "raw_file": str(s)} for s in sessions]
    else:
        rows = [dict(s) for s in sessions]

    for row in rows:
        for key, value in list(row.items()):
            if value in ("", None):
                row.pop(key)
        if "id" not in row:
            if "raw_file" in row:
                row["id"] = find_run_id(str(row["raw_file"]))
            else:
                raise ValueError(
                    f"A session has no 'id' and no 'raw_file' to derive it from: {row}"
                )
    ids = [row["id"] for row in rows]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Session ids must be unique; duplicated: {duplicates}.")
    return rows


def columns(sessions: list[dict]) -> set:
    """The columns any session has."""
    return {key for row in sessions for key in row}


def by_subject(sessions: list[dict]) -> OrderedDict:
    """Sessions grouped by subject, in order of first appearance.

    A session with no ``subject`` is its own subject (its id).
    """
    groups = OrderedDict()
    for row in sessions:
        groups.setdefault(row.get("subject", row["id"]), []).append(row)
    return groups


def output_paths(session: dict, outdir) -> dict:
    """Where a session's outputs go under ``outdir`` (the osl-dynamics layout).

    Returns
    -------
    paths : dict
        ``preproc_file`` (the manifest's if given), ``preproc_dir``,
        ``surfaces_dir`` (per subject), ``session_dir`` and ``parc_file``.
    """
    outdir = Path(outdir)
    id_ = session["id"]
    subject = session.get("subject", id_)
    return {
        "preproc_dir": outdir / "preproc",
        "preproc_file": (
            Path(session["preproc_file"])
            if "preproc_file" in session
            else outdir / "preproc" / f"{id_}_preproc-raw.fif"
        ),
        "surfaces_dir": outdir / "surfaces" / subject,
        "session_dir": outdir / id_,
        "parc_file": outdir / id_ / f"{id_}_parc-raw.fif",
    }
