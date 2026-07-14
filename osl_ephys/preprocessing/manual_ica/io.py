"""Plain-text I/O for the labels and bad-segment files written by the
review server.

These are the canonical parsers --- if you have your own pipeline that needs
to act on the user's review decisions, import from here instead of
re-implementing the regex (semp's ``2.ica.py`` previously had its own copy
that drifted; using the package's parser keeps things in sync).
"""
import re
from pathlib import Path

__all__ = ['parse_label_txt', 'parse_bads_txt', 'LABEL_LINE_RX']


# Canonical line: "ICnnn: bad" / "ICnnn: good" / "ICnnn: unsure" /
# "ICnnn: unlabeled". Comments (#-prefixed) and the trailing
# "bad_ics: [..]" footer are tolerated by skipping them in the parser.
LABEL_LINE_RX = re.compile(r'^IC(\d+):\s*(bad|good|unsure|unlabeled)\s*$')


def parse_label_txt(path):
    """Parse a ``label.txt`` written by the review server.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    bad_ics    : list[int]   --- sorted, unique ICs marked ``bad``
    n_unsure   : int         --- count of ICs marked ``unsure`` (kept, not removed)
    n_unlabeled: int         --- count of ICs marked ``unlabeled`` (review unfinished)
    warnings   : list[str]   --- one entry per unparsable IC-looking line
                                  (e.g. typo in the state name); empty if clean
    """
    bad, n_unsure, n_unlabeled = [], 0, 0
    warnings = []
    with open(path) as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith('#') or line.startswith('bad_ics'):
                continue
            m = LABEL_LINE_RX.match(line)
            if not m:
                if line.upper().startswith('IC'):
                    warnings.append(f'line {lineno}: unparsable {line!r}')
                continue
            idx, state = int(m.group(1)), m.group(2)
            if state == 'bad':
                bad.append(idx)
            elif state == 'unsure':
                n_unsure += 1
            elif state == 'unlabeled':
                n_unlabeled += 1
    return sorted(set(bad)), n_unsure, n_unlabeled, warnings


def parse_bads_txt(path):
    """Parse a ``bads.txt`` written by the review server's B-modal.

    Returns a list of ``(onset_s, duration_s)`` pairs (recording-relative
    seconds) suitable for appending to ``raw.annotations`` as
    ``BAD_manual``. Blank lines, ``#`` comments, and lines without two
    parseable floats are silently skipped --- the file is small and
    user-edited, so a malformed line is more likely "intentional comment"
    than "lost data".
    """
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                start, end = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if end > start:
                out.append((start, end - start))
    return out
