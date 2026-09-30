"""Default user-facing configuration for ``manual_ica``.

Kept as a module-level dict (instead of a function-local one) so it's easy
to inspect and override from a project script::

    from osl_ephys.preprocessing.manual_ica import config
    custom = {**config.DEFAULT_USERARGS, 'n_components': 50}

Any key supplied to ``manual_ica``'s ``userargs`` that is *not* in
``DEFAULT_USERARGS`` is rejected (proc_userargs strict mode), so this dict
is also the canonical schema.
"""

DEFAULT_USERARGS = {
    # ── ICA fit ────────────────────────────────────────────────────────────
    'n_components':  0.999,
    'method':        'fastica',
    'max_iter':      'auto',
    'seed':          42,
    'l_freq':        1.0,
    'picks':         'eeg',

    # ── auto-flagging channels & thresholds ───────────────────────────────
    'ecg_ch':        None,
    'eog_ch':        None,
    # Sampling-rate-dependent CTPS threshold, as in OSL's ica_autoreject.
    # EEG-fMRI examples explicitly use 0.1 for residual pulse artifacts.
    'ecg_threshold': 'auto',
    # Absolute Pearson correlation, matching OSL's ica_autoreject default.
    'eog_threshold': 0.35,
    # Gate 1: slice-harmonic peak / median local shoulder PSD.
    'ga_local_threshold': 8.0,
    # Gate 2: slice-harmonic peak / strongest other 5-45 Hz peak.
    'ga_dominance_threshold': 1.0,
    'ga_fmin': 1.0,
    'ga_fmax': 45.0,
    'ga_dominance_fmin': 5.0,
    'ga_peak_window': 1.0,
    'ga_base_window': 5.0,

    # ── plotting ───────────────────────────────────────────────────────────
    'latex_mode':     False,
    'seg_len':        None,         # None -> 2.0 s (matches bad_segments)
    'spec_type':      'linear',     # 'linear' | 'db'
    'spec_freq_max':  45.0,
    'psd_resolution': 0.05,
    'zoom_window':    20.0,

    # ── output ─────────────────────────────────────────────────────────────
    # IC review root. Per-subject subdirs are created under this. Safe to
    # set in batch userargs because it's the *root*, not subject-keyed. If
    # None, falls back to dataset['target_pth']/'ica', then cwd.
    # NB: 'subject' is *deliberately* NOT a userarg (proc_userargs strict
    # mode will reject it) --- run_proc_batch shares one userargs dict
    # across all subjects, so a userargs subject would clobber every
    # subject's review folder. Set dataset['subject'] in an upstream
    # extra_func instead.
    'outdir':        None,
}
