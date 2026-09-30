"""Plain-text I/O for the labels and bad-segment files written by the
review server.

These are the canonical parsers --- if you have your own pipeline that needs
to act on the user's review decisions, import from here instead of
re-implementing the regex (semp's ``2.ica.py`` previously had its own copy
that drifted; using the package's parser keeps things in sync).
"""
import math
import re
from pathlib import Path

__all__ = ['parse_label_txt', 'parse_bads_txt', 'LABEL_LINE_RX']


# Canonical line: "ICnnn: bad" / "ICnnn: good" / "ICnnn: unsure" /
# "ICnnn: unlabeled". Comments (#-prefixed) and the trailing
# "bad_ics: [..]" footer are tolerated by skipping them in the parser.
LABEL_LINE_RX = re.compile(r'^IC(\d+):\s*(bad|good|unsure|unlabeled)\s*$')


def parse_label_txt(path, expected_components=None):
    """Parse a ``label.txt`` written by the review server.

    Parameters
    ----------
    path : str | Path
    expected_components : int | None
        If given, require exactly one label for every IC index from zero to
        ``expected_components - 1``.

    Returns
    -------
    bad_ics    : list[int]   --- sorted, unique ICs marked ``bad``
    n_unsure   : int         --- count of ICs marked ``unsure`` (kept, not removed)
    n_unlabeled: int         --- count of ICs marked ``unlabeled`` (review unfinished)
    warnings   : list[str]   --- malformed, duplicate, missing, or extra labels
    """
    labels = {}
    warnings = []
    with open(path) as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith('#') or line.startswith('bad_ics'):
                continue
            m = LABEL_LINE_RX.match(line)
            if not m:
                warnings.append(f'line {lineno}: unparsable {line!r}')
                continue
            idx, state = int(m.group(1)), m.group(2)
            if idx in labels:
                warnings.append(f'line {lineno}: duplicate IC{idx:03d} label')
                continue
            labels[idx] = state

    if expected_components is not None:
        expected = set(range(expected_components))
        missing = sorted(expected - labels.keys())
        extra = sorted(labels.keys() - expected)
        if missing:
            warnings.append(f'missing IC labels: {missing}')
        if extra:
            warnings.append(f'out-of-range IC labels: {extra}')

    bad = sorted(idx for idx, state in labels.items() if state == 'bad')
    n_unsure = sum(state == 'unsure' for state in labels.values())
    n_unlabeled = sum(state == 'unlabeled' for state in labels.values())
    return bad, n_unsure, n_unlabeled, warnings


def parse_bads_txt(path):
    """Parse a ``bads.txt`` written by the review server's B-modal.

    Returns a list of ``(onset_s, duration_s)`` pairs (recording-relative
    seconds) suitable for appending to ``raw.annotations`` as
    ``BAD_manual``. Blank lines and ``#`` comments are allowed. Malformed or
    non-positive intervals raise with their line number rather than silently
    discarding a reviewed bad segment.
    """
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p) as f:
        for lineno, raw_line in enumerate(f, 1):
            line = raw_line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f'{p}:{lineno}: expected start and end seconds')
            try:
                start, end = float(parts[0]), float(parts[1])
            except ValueError as exc:
                raise ValueError(f'{p}:{lineno}: invalid start or end seconds') from exc
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError(f'{p}:{lineno}: require finite 0 <= start < end')
            out.append((start, end - start))
    return out
