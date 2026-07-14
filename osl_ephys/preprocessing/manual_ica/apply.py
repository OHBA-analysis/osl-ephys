"""Apply manual-ICA review decisions to preprocessed raw fifs.

Reads the per-subject ``label.txt`` (IC labels) and ``bads.txt`` (manual
bad time segments) written by the review server, then applies the saved
ICA decomposition with the flagged ICs in ``ica.exclude`` and the manual
bad segments added to ``raw.annotations`` as ``BAD_manual``.

This is the standalone post-review step that complements ``manual_ica``
(which only fits + renders the review pages during the preproc batch).
The split exists because the labels are produced by a human between the
two runs.

Command line
------------

    osl-ica-apply <ica_root> <raw_root> [subject ...] [--overwrite]

Paths
-----

For each ``<subject>``::

    raw_path   = <raw_root>/<subject>/<subject>_preproc-raw.fif
    ica_path   = <raw_root>/<subject>/<subject>_ica.fif
    label_path = <ica_root>/<subject>/label.txt
    bads_path  = <ica_root>/<subject>/bads.txt
    out_path   = <raw_root>/<subject>/<subject>_after_ica-raw.fif

If no subject is given, every directory under ``<ica_root>`` that contains
a ``label.txt`` is processed.
"""
import argparse
import sys
import time
from pathlib import Path

import mne

from .io import parse_label_txt, parse_bads_txt


RAW_SUFFIX = '_preproc-raw.fif'
ICA_SUFFIX = '_ica.fif'
OUT_SUFFIX = '_after_ica-raw.fif'


def _purge_review_assets(subject_dir):
    """Delete per-IC SVG renders + the HTML review pages for one subject.

    Keeps ``label.txt`` and ``bads.txt`` (the user's decisions --- cheap to
    keep, expensive to redo). Returns ``(n_removed, freed_bytes)``.
    """
    n_removed = 0
    freed = 0
    for pattern in ('ic_*.svg', 'topo_*.svg',
                    'single_ic.html', 'between_ic.html'):
        for p in subject_dir.glob(pattern):
            try:
                freed += p.stat().st_size
                p.unlink()
                n_removed += 1
            except OSError:
                pass
    for sub in ('zoomed', 'zoomed_clean'):
        d = subject_dir / sub
        if d.is_dir():
            for p in d.glob('*.svg'):
                try:
                    freed += p.stat().st_size
                    p.unlink()
                    n_removed += 1
                except OSError:
                    pass
            try:
                d.rmdir()
            except OSError:
                pass
    return n_removed, freed


def apply_one(ica_root, raw_root, subject, overwrite=False, purge_svgs=False):
    """Apply review decisions for one subject.

    Parameters
    ----------
    ica_root : path-like
        Root containing ``<subject>/label.txt`` (and ``bads.txt``).
    raw_root : path-like
        Root containing ``<subject>/<subject>_preproc-raw.fif`` and
        ``<subject>/<subject>_ica.fif``.
    subject : str
    overwrite : bool
        Overwrite ``<subject>_after_ica-raw.fif`` if it exists.
    purge_svgs : bool
        After a successful apply, delete the per-IC SVGs + HTML review
        pages under ``<ica_root>/<subject>/`` (keeps label.txt / bads.txt).
        On a typical Staresina subject this frees ~50-200 MB.

    Returns a status string (``ok ...`` or ``skip ...`` / ``error ...``)
    suitable for printing in batch mode.
    """
    ica_root = Path(ica_root)
    raw_root = Path(raw_root)
    raw_path   = raw_root / subject / f'{subject}{RAW_SUFFIX}'
    ica_path   = raw_root / subject / f'{subject}{ICA_SUFFIX}'
    label_path = ica_root / subject / 'label.txt'
    bads_path  = ica_root / subject / 'bads.txt'
    out_path   = raw_root / subject / f'{subject}{OUT_SUFFIX}'

    missing = [p for p in (raw_path, ica_path, label_path) if not p.exists()]
    if missing:
        return f'skip --- missing: {[p.name for p in missing]}'
    if out_path.exists() and not overwrite:
        return f'skip --- exists: {out_path.name}'

    bad_ics, n_unsure, n_unlabeled, warnings = parse_label_txt(label_path)
    if warnings:
        return (f'skip --- {len(warnings)} bad lines in label.txt: '
                + ' | '.join(warnings[:3])
                + ('...' if len(warnings) > 3 else ''))
    if n_unlabeled:
        return f'skip --- {n_unlabeled} unlabeled ICs (review not finished)'

    bad_segs = parse_bads_txt(bads_path) if bads_path.exists() else []

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
    if bad_segs:
        raw.annotations.append(
            [s for s, _ in bad_segs],
            [d for _, d in bad_segs],
            ['BAD_manual'] * len(bad_segs),
        )
    ica = mne.preprocessing.read_ica(ica_path, verbose=False)
    out_of_range = [i for i in bad_ics if i >= ica.n_components_]
    if out_of_range:
        return (f'skip --- label.txt references IC indices >= '
                f'n_components_={ica.n_components_}: {out_of_range}')
    ica.exclude = list(bad_ics)
    cleaned = ica.apply(raw, verbose=False)
    cleaned.save(out_path, overwrite=overwrite, verbose=False)
    msg = (f'ok --- removed {len(bad_ics)} bad / kept '
           f'{ica.n_components_ - len(bad_ics)} '
           f'(unsure={n_unsure}, bad segs={len(bad_segs)}) -> {out_path.name}')
    if purge_svgs:
        n_removed, freed = _purge_review_assets(ica_root / subject)
        msg += f' [+ purged {n_removed} review files, freed {freed / 1e6:.1f} MB]'
    return msg


def main():
    p = argparse.ArgumentParser(
        description='Apply manual ICA review decisions to preprocessed fifs.'
    )
    p.add_argument('ica_root', help='Root containing <subject>/label.txt')
    p.add_argument('raw_root', help='Root containing <subject>/<subject>_preproc-raw.fif and _ica.fif')
    p.add_argument('subjects', nargs='*',
                   help='Subjects to process. Default: every subject under '
                        'ica_root with a label.txt.')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing <subject>_after_ica-raw.fif.')
    p.add_argument('--purge-svgs', action='store_true',
                   help='After a successful apply, delete the per-IC SVGs '
                        'and HTML review pages under <ica_root>/<subject>/ '
                        'to save disk space (label.txt + bads.txt are kept).')
    args = p.parse_args()

    ica_root = Path(args.ica_root)
    if not ica_root.exists():
        sys.exit(f'ICA folder not found: {ica_root}')

    subjects = args.subjects or sorted(
        d.name for d in ica_root.iterdir()
        if d.is_dir() and (d / 'label.txt').exists()
    )
    print(f'[osl-ica-apply] {len(subjects)} subject(s)')

    n_done = n_skip = 0
    t0 = time.time()
    for s in subjects:
        try:
            msg = apply_one(ica_root, args.raw_root, s,
                            overwrite=args.overwrite,
                            purge_svgs=args.purge_svgs)
        except Exception as e:
            msg = f'error --- {type(e).__name__}: {e}'
        print(f'  {s}: {msg}')
        if msg.startswith('ok'):
            n_done += 1
        else:
            n_skip += 1
    print(f'[osl-ica-apply] {n_done} processed, {n_skip} skipped '
          f'in {time.time() - t0:.1f} s')


if __name__ == '__main__':
    main()
