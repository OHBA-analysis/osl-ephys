"""Signal-/dataset-level helpers used by manual_ica.

Nothing in this module touches matplotlib --- those primitives live in
``vis.py``. Anything HTML-rendering-related lives in ``html.py``.
"""
from pathlib import Path

import numpy as np
import mne

from ...utils import mean_psd_in_band


# ── annotations -------------------------------------------------------------

def _good_mask(raw):
    """Boolean mask of length ``raw.n_times``: True iff sample is NOT inside
    any annotation whose description starts with ``'bad'``
    (case-insensitive)."""
    n = raw.n_times
    sfreq = raw.info['sfreq']
    first_time = raw.first_samp / sfreq
    mask = np.ones(n, dtype=bool)
    for ann in raw.annotations:
        if str(ann['description']).lower().startswith('bad'):
            i0 = max(0, int(round((ann['onset'] - first_time) * sfreq)))
            i1 = min(n, int(round((ann['onset'] + ann['duration'] - first_time) * sfreq)))
            if i1 > i0:
                mask[i0:i1] = False
    return mask


# ── channel selection ------------------------------------------------------

def _safe_chname(name):
    """Filename-safe channel name (alnum or underscore)."""
    return ''.join(c if c.isalnum() else '_' for c in str(name))


def _pick_supp_channels(raw):
    """List of (display_name, file_safe_name) for ECG/EOG/EMG channels."""
    types = raw.get_channel_types()
    out = []
    for i, t in enumerate(types):
        if t in ('ecg', 'eog', 'emg'):
            name = raw.ch_names[i]
            out.append((name, _safe_chname(name)))
    return out


# ── scoring ----------------------------------------------------------------

def _compute_slice_ratios(src_data, sfreq, slice_interval, tr_interval,
                          good_mask=None, fmin=1.0,
                          noise_window=1.0, base_window=5.0):
    """Per-IC max residual gradient artifact ratio (noise / base) at slice
    harmonics --- same metric as ``slice_ica``, applied to each IC.

    If ``good_mask`` is given, only those samples are used (bad-annotation
    samples excluded --- same convention as the rest of ``manual_ica``).
    """
    eps = 1e-10
    slice_freq = 1.0 / slice_interval
    tr_freq    = 1.0 / tr_interval
    # cover at least 3 slice harmonics, capped at Nyquist - guard band.
    fmax = min(sfreq / 2.0 - max(2.0, base_window * tr_freq), 3.5 * slice_freq)
    if fmax <= slice_freq:
        return np.zeros(src_data.shape[0])
    data_for_psd = src_data if good_mask is None else src_data[:, good_mask]
    psds, freqs = mne.time_frequency.psd_array_welch(
        data_for_psd, sfreq=sfreq, fmin=fmin, fmax=fmax,
        n_fft=int(round(sfreq * 20)), verbose=False,
    )
    harmonics = np.arange(slice_freq, freqs.max(), slice_freq)
    ratios = np.zeros(src_data.shape[0])
    for ic in range(src_data.shape[0]):
        psd_row = psds[ic]
        for harmonic in harmonics:
            noise = mean_psd_in_band(psd_row, freqs, harmonic, noise_window * tr_freq / 2)
            base  = mean_psd_in_band(psd_row, freqs, harmonic, base_window  * tr_freq / 2)
            base  = (base * base_window - noise * noise_window) / (base_window - noise_window)
            r = noise / (base + eps)
            if r > ratios[ic]:
                ratios[ic] = r
    return ratios


def _build_scores_list(n_components, ecg_scores, ecg_idx_auto, ecg_threshold,
                       eog_scores_list, eog_idx_auto, eog_threshold,
                       slice_ratios=None, ga_threshold=4.0):
    """Build per-IC scores dicts for embedding in the review HTML.

    Each dict has at most:
      ecg, ecg_thr, ecg_flag      --- scalar ECG CTPS
      eog: [{ch, val, thr, flag}] --- one entry per EOG channel
      ga,  ga_thr,  ga_flag       --- residual GA ratio (slice harmonics)

    Flags compare ``|value| > threshold`` directly (do NOT rely on MNE's
    ``find_bads_*`` index lists, which use their own internal thresholds
    --- e.g. ctps default 0.25 != user's 0.1).
    """
    out = []
    for i in range(n_components):
        entry = {}
        if ecg_scores is not None:
            v = float(ecg_scores[i])
            entry['ecg'] = round(v, 4)
            entry['ecg_thr'] = ecg_threshold
            entry['ecg_flag'] = abs(v) > ecg_threshold
        eog_items = []
        for ch_name, ch_scores in (eog_scores_list or []):
            v = float(ch_scores[i])
            eog_items.append({
                'ch':   ch_name,
                'val':  round(v, 4),
                'thr':  eog_threshold,
                'flag': abs(v) > eog_threshold,
            })
        entry['eog'] = eog_items
        if slice_ratios is not None:
            r = float(slice_ratios[i])
            entry['ga']      = round(r, 3)
            entry['ga_thr']  = ga_threshold
            entry['ga_flag'] = r > ga_threshold
        out.append(entry)
    return out


# ── dataset/userargs resolution -------------------------------------------

def _resolve_subject_and_outdir(dataset, userargs):
    """Pick (subject, ica_outdir) from ``dataset['subject']`` and userargs.

    The subject ID **must** come from ``dataset['subject']`` --- never from
    userargs and never auto-derived. Rationale: osl-ephys's
    ``run_proc_batch`` shares a single config (and therefore a single
    ``userargs`` dict) across every subject in the batch, so a userargs
    ``subject`` would silently overwrite every subject's review folder
    with the first subject's ID. The caller is expected to add
    ``dataset['subject']`` in an upstream extra_func (e.g. semp's
    ``initialize`` step does this).

    ``outdir`` is genuinely batch-constant (it's the IC *root*, with a
    per-subject subdir created underneath), so it's safe to take from
    userargs. Resolution order for outdir:
      1. ``userargs['outdir']``
      2. ``dataset['target_pth'] / 'ica'``  (semp convention)
      3. ``Path.cwd() / 'manual_ica_review'``  (last-resort default)
    """
    subject = dataset.get('subject')
    if subject is None:
        raise KeyError(
            "manual_ica needs dataset['subject'] to be set by an upstream "
            "extra_func. Do not pass 'subject' via userargs --- the same "
            "userargs dict is shared across every subject in a "
            "run_proc_batch, so a userargs subject would clobber every "
            "subject's review folder."
        )

    if userargs.get('outdir') is not None:
        ica_root = Path(userargs['outdir'])
    elif dataset.get('target_pth') is not None:
        ica_root = Path(dataset['target_pth']) / 'ica'
    else:
        ica_root = Path.cwd() / 'manual_ica_review'

    return subject, ica_root / subject
