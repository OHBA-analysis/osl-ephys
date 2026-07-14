"""Manual ICA review for osl-ephys preprocessing pipelines.

This subpackage adds the ``manual_ica`` preprocessing wrapper, which fits
an ICA decomposition and writes a per-subject HTML review page. The page
lets the user click through ICs and label them ``good`` / ``bad`` /
``unsure`` (plus mark bad time segments) entirely in a browser; results
are written back to ``label.txt`` / ``bads.txt`` by the bundled review
server (``osl-ica-review``) and applied via ``osl-ica-apply``.

Because ``manual_ica`` is now resolved through osl-ephys's
``find_func`` (``run_osl_manual_ica`` in ``osl_wrappers``), an
osl-ephys config can use it directly without passing it via
``extra_funcs``::

    config = {'preproc': [
        ...,
        {'manual_ica': {'n_components': 0.999, 'picks': 'eeg',
                        'outdir': '/path/to/ica_review'}},
    ]}

``dataset['subject']`` must be set by an upstream extra_func.

To review the results, run the bundled HTTP server inside the IC root::

    osl-ica-review 8000
    # then open http://localhost:8000/<subject>/single_ic.html

To apply the review decisions::

    osl-ica-apply <ica_root> <raw_root> <subject>
"""
from .ica import manual_ica, apply_ica
from .io import parse_label_txt, parse_bads_txt
from .apply import apply_one as apply_manual_ica

__all__ = [
    'manual_ica',
    'apply_ica',
    'parse_label_txt',
    'parse_bads_txt',
    'apply_manual_ica',
]
