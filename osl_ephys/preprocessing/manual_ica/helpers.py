"""Signal-/dataset-level helpers used by manual_ica.

Nothing in this module touches matplotlib --- those primitives live in
``vis.py``. Anything HTML-rendering-related lives in ``html.py``.
"""
from pathlib import Path

import numpy as np
import mne

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

def _slice_peak_to_base(psd_row, freqs, harmonic, peak_half_width,
                        base_half_width):
    """Peak PSD near one harmonic divided by its local shoulder median.

    A maximum is intentional: residual gradient artefact can occupy only one
    or two 0.05-Hz bins and is visually unacceptable even when its average
    across a wider band is modest. The median shoulder is robust to the peak
    itself and to isolated neighboring lines.
    """
    distance = np.abs(freqs - harmonic)
    peak_mask = distance <= peak_half_width
    base_mask = (distance > peak_half_width) & (distance <= base_half_width)
    if not np.any(peak_mask) or not np.any(base_mask):
        return 0.0

    peak = float(np.max(psd_row[peak_mask]))
    base = float(np.median(psd_row[base_mask]))
    if base <= 0:
        return np.inf if peak > 0 else 0.0
    return peak / base


def _compute_slice_ga_scores(src_data, sfreq, slice_interval, tr_interval,
                             good_mask=None, fmin=1.0, fmax=45.0,
                             dominance_fmin=5.0,
                             peak_window=1.0, base_window=5.0,
                             local_threshold=8.0,
                             dominance_threshold=1.0):
    """Return complementary local-prominence and global-dominance GA scores.

    If ``good_mask`` is given, only those samples are used (bad-annotation
    samples excluded --- same convention as the rest of ``manual_ica``).

    ``peak_window`` and ``base_window`` are full widths in multiples of the TR
    frequency. The defaults search within +/-0.5/TR Hz of each harmonic and
    estimate background from the surrounding region out to +/-2.5/TR Hz.

    The second score asks whether that same harmonic peak also exceeds every
    non-harmonic peak between ``dominance_fmin`` and ``fmax``. Each harmonic
    is tested as a pair; an IC passes only when one harmonic passes both gates.
    This prevents a locally sharp peak at one harmonic and a globally dominant
    but locally broad peak at another harmonic from being combined.
    """
    if slice_interval <= 0 or tr_interval <= 0:
        raise ValueError('slice_interval and tr_interval must be positive.')
    if peak_window <= 0 or base_window <= peak_window:
        raise ValueError('base_window must be greater than peak_window > 0.')
    if fmax <= dominance_fmin:
        raise ValueError('fmax must be greater than dominance_fmin.')
    if local_threshold <= 0 or dominance_threshold <= 0:
        raise ValueError('GA thresholds must be positive.')
    slice_freq = 1.0 / slice_interval
    tr_freq = 1.0 / tr_interval
    analysis_fmax = min(
        float(fmax), sfreq / 2.0 - max(2.0, base_window * tr_freq)
    )
    empty = {
        'peak_to_local': np.zeros(src_data.shape[0]),
        'peak_to_elsewhere': np.zeros(src_data.shape[0]),
        'harmonic_hz': np.zeros(src_data.shape[0]),
        'harmonics_hz': np.zeros(0),
        'local_by_harmonic': np.zeros((src_data.shape[0], 0)),
        'peak_power_by_harmonic': np.zeros((src_data.shape[0], 0)),
        'base_power_by_harmonic': np.zeros((src_data.shape[0], 0)),
    }
    if analysis_fmax <= slice_freq:
        return empty
    data_for_psd = src_data if good_mask is None else src_data[:, good_mask]
    if data_for_psd.shape[1] < 2:
        return empty
    psds, freqs = mne.time_frequency.psd_array_welch(
        data_for_psd, sfreq=sfreq, fmin=fmin, fmax=analysis_fmax,
        n_fft=min(int(round(sfreq * 20)), data_for_psd.shape[1]),
        verbose=False,
    )
    harmonics = np.arange(slice_freq, freqs.max() + 1e-12, slice_freq)
    local_by_harmonic = np.zeros((src_data.shape[0], len(harmonics)))
    dominance_by_harmonic = np.zeros((src_data.shape[0], len(harmonics)))
    peak_power_by_harmonic = np.zeros((src_data.shape[0], len(harmonics)))
    base_power_by_harmonic = np.zeros((src_data.shape[0], len(harmonics)))
    peak_half_width = peak_window * tr_freq / 2
    base_half_width = base_window * tr_freq / 2
    harmonic_mask = np.zeros(len(freqs), dtype=bool)
    for harmonic in harmonics:
        harmonic_mask |= np.abs(freqs - harmonic) <= peak_half_width
    elsewhere_mask = (freqs >= dominance_fmin) & ~harmonic_mask
    if not np.any(harmonic_mask) or not np.any(elsewhere_mask):
        return empty

    for ic in range(src_data.shape[0]):
        psd_row = psds[ic]
        elsewhere_peak = float(np.max(psd_row[elsewhere_mask]))
        for harmonic_index, harmonic in enumerate(harmonics):
            distance = np.abs(freqs - harmonic)
            peak_mask = np.abs(freqs - harmonic) <= peak_half_width
            base_mask = (
                (distance > peak_half_width)
                & (distance <= base_half_width)
            )
            harmonic_peak = float(np.max(psd_row[peak_mask]))
            harmonic_base = float(np.median(psd_row[base_mask]))
            peak_power_by_harmonic[ic, harmonic_index] = harmonic_peak
            base_power_by_harmonic[ic, harmonic_index] = harmonic_base
            if harmonic_base <= 0:
                local_by_harmonic[ic, harmonic_index] = (
                    np.inf if harmonic_peak > 0 else 0.0
                )
            else:
                local_by_harmonic[ic, harmonic_index] = (
                    harmonic_peak / harmonic_base
                )
            if elsewhere_peak <= 0:
                dominance_by_harmonic[ic, harmonic_index] = (
                    np.inf if harmonic_peak > 0 else 0.0
                )
            else:
                dominance_by_harmonic[ic, harmonic_index] = (
                    harmonic_peak / elsewhere_peak
                )

    # Select the harmonic closest to jointly passing both thresholds.  Taking
    # the minimum normalized margin makes ``best_margin > 1`` exactly
    # equivalent to one harmonic passing both gates.
    joint_margin = np.minimum(
        local_by_harmonic / local_threshold,
        dominance_by_harmonic / dominance_threshold,
    )
    best_harmonic = np.argmax(joint_margin, axis=1)
    component_index = np.arange(src_data.shape[0])
    return {
        'peak_to_local': local_by_harmonic[component_index, best_harmonic],
        'peak_to_elsewhere': dominance_by_harmonic[
            component_index, best_harmonic
        ],
        'harmonic_hz': harmonics[best_harmonic],
        # The matrices below retain the per-harmonic information needed by
        # optional downstream gates. The three historical one-dimensional
        # scores above remain unchanged.
        'harmonics_hz': harmonics,
        'local_by_harmonic': local_by_harmonic,
        'peak_power_by_harmonic': peak_power_by_harmonic,
        'base_power_by_harmonic': base_power_by_harmonic,
    }


def _compute_slice_contribution_scores(scores, component_maps, excluded=(),
                                       local_threshold=4.0,
                                       fraction_threshold=0.30):
    """Score each IC's share of the remaining sensor-level harmonic excess.

    For each component and slice harmonic, the scale-invariant contribution is
    ``max(source_peak - source_background, 0) * mean(component_map ** 2)``.
    Fractions are calculated after removing components already scheduled for
    rejection, so the score targets residual rather than already-handled GA.

    This is an optional rescue gate, not a replacement for the conservative
    local-prominence/global-dominance gate.
    """
    if local_threshold <= 0:
        raise ValueError('contribution local_threshold must be positive.')
    if not 0 < fraction_threshold <= 1:
        raise ValueError(
            'contribution fraction_threshold must be in the interval (0, 1].'
        )

    local = np.asarray(scores['local_by_harmonic'], dtype=float)
    peak = np.asarray(scores['peak_power_by_harmonic'], dtype=float)
    base = np.asarray(scores['base_power_by_harmonic'], dtype=float)
    harmonics = np.asarray(scores['harmonics_hz'], dtype=float)
    maps = np.asarray(component_maps, dtype=float)
    if local.shape != peak.shape or local.shape != base.shape:
        raise ValueError('slice score matrices must have identical shapes.')
    if maps.ndim != 2 or maps.shape[1] != local.shape[0]:
        raise ValueError(
            'component_maps must have shape (n_channels, n_components).'
        )

    map_power = np.mean(maps ** 2, axis=0)
    excess = np.maximum(peak - base, 0.0) * map_power[:, np.newaxis]
    remaining = np.ones(local.shape[0], dtype=bool)
    excluded = np.asarray(list(excluded), dtype=int)
    if excluded.size:
        if np.any(excluded < 0) or np.any(excluded >= local.shape[0]):
            raise ValueError('excluded contains an out-of-range ICA index.')
        remaining[excluded] = False

    denominator = np.sum(excess[remaining], axis=0)
    fractions = np.zeros_like(excess)
    np.divide(
        excess,
        denominator[np.newaxis, :],
        out=fractions,
        where=denominator[np.newaxis, :] > 0,
    )
    passes = (
        remaining[:, np.newaxis]
        & (local > local_threshold)
        & (fractions > fraction_threshold)
    )
    eligible_fraction = np.where(
        remaining[:, np.newaxis] & (local > local_threshold),
        fractions,
        -1.0,
    )
    best = np.argmax(eligible_fraction, axis=1) if fractions.shape[1] else np.zeros(
        local.shape[0], dtype=int
    )
    components = np.arange(local.shape[0])
    if fractions.shape[1]:
        best_fraction = fractions[components, best]
        best_local = local[components, best]
        best_harmonic = harmonics[best]
    else:
        best_fraction = np.zeros(local.shape[0])
        best_local = np.zeros(local.shape[0])
        best_harmonic = np.zeros(local.shape[0])
    return {
        'fraction': best_fraction,
        'local': best_local,
        'harmonic_hz': best_harmonic,
        'passes': np.any(passes, axis=1),
        'fraction_by_harmonic': fractions,
    }


def _build_scores_list(n_components, ecg_scores, ecg_idx_auto, ecg_threshold,
                       eog_scores_list, eog_idx_auto, eog_threshold,
                       slice_scores=None, ga_local_threshold=8.0,
                       ga_dominance_threshold=1.0):
    """Build per-IC scores dicts for embedding in the review HTML.

    Each dict has at most:
      ecg, ecg_thr, ecg_flag      --- scalar ECG CTPS
      eog: [{ch, val, thr, flag}] --- one entry per EOG channel
      ga_local, ga_dominance, thresholds, ga_flag --- two-gate GA score

    Flags compare scores directly with the displayed thresholds: CTPS uses
    ``>=`` like MNE, and EOG uses ``|value| > threshold`` (do NOT rely on MNE's
    ``find_bads_*`` index lists, which use their own internal thresholds
    --- e.g. CTPS auto is 0.32 at 250 Hz, while EEG-fMRI may use 0.1).
    """
    out = []
    for i in range(n_components):
        entry = {}
        if ecg_scores is not None:
            v = float(ecg_scores[i])
            entry['ecg'] = round(v, 4)
            entry['ecg_thr'] = ecg_threshold
            entry['ecg_flag'] = abs(v) >= ecg_threshold
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
        if slice_scores is not None:
            local = float(slice_scores['peak_to_local'][i])
            dominance = float(slice_scores['peak_to_elsewhere'][i])
            entry['ga_local'] = round(local, 3)
            entry['ga_local_thr'] = ga_local_threshold
            entry['ga_dominance'] = round(dominance, 3)
            entry['ga_dominance_thr'] = ga_dominance_threshold
            entry['ga_flag'] = (
                local > ga_local_threshold
                and dominance > ga_dominance_threshold
            )
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
