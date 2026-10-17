"""Where a run keeps a session's data, ICA and review files."""

from __future__ import annotations

from pathlib import Path


def session_files(outdir, id):
    """The preprocessed file of a session and its QC directory."""
    outdir = Path(outdir)
    preproc_file = outdir / "preproc" / f"{id}_preproc-raw.fif"
    if not preproc_file.exists():
        raise FileNotFoundError(f"No preprocessed data for '{id}': {preproc_file}")
    return preproc_file, outdir / "qc"


def ica_files(outdir, id):
    """The session's ICA files, in the order they were fitted."""
    preproc_dir = Path(outdir) / "preproc"
    files = [preproc_dir / f"{id}_ica.fif"]
    n = 2
    while (preproc_dir / f"{id}_ica-{n}.fif").exists():
        files.append(preproc_dir / f"{id}_ica-{n}.fif")
        n += 1
    if not files[0].exists():
        raise FileNotFoundError(f"No ICA for '{id}': {files[0]}")
    return files


def review_files(outdir, id):
    """The browser review's label and bad segment files, if that route was used."""
    root = Path(outdir) / "ica" / id
    return (
        (root / "label.txt", root / "bads.txt")
        if (root / "label.txt").exists()
        else None
    )


def sessions_with_ica(outdir):
    """The ids of the sessions of a run that have an ICA."""
    preproc_dir = Path(outdir) / "preproc"
    return sorted(p.name[: -len("_ica.fif")] for p in preproc_dir.glob("*_ica.fif"))
