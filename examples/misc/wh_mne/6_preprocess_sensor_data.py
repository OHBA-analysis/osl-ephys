"""Preprocess the sensor-level data.

"""

import os
import mne
import numpy as np
from scipy import stats

def get_bad_indices(x, window_length, significance_level=0.05, maximum_fraction=0.1):
    """Automated bad segment detection using the G-ESD algorithm."""
    def _gesd(X, alpha=significance_level, p_out=maximum_fraction, outlier_side=0):
        if outlier_side == 0:
            alpha = alpha / 2
        n_out = int(np.ceil(len(X) * p_out))
        if np.any(np.isnan(X)):
            y = np.where(np.isnan(X))[0]
            idx1, x2 = _gesd(X[np.isfinite(X)], alpha, n_out, outlier_side)
            idx = np.zeros_like(X).astype(bool)
            idx[y[idx1]] = True
        n = len(X)
        temp = X.copy()
        R = np.zeros(n_out)
        rm_idx = np.zeros(n_out, dtype=int)
        lam = np.zeros(n_out)
        for j in range(0, int(n_out)):
            i = j + 1
            if outlier_side == -1:
                rm_idx[j] = np.nanargmin(temp)
                sample = np.nanmin(temp)
                R[j] = np.nanmean(temp) - sample
            elif outlier_side == 0:
                rm_idx[j] = int(np.nanargmax(abs(temp - np.nanmean(temp))))
                R[j] = np.nanmax(abs(temp - np.nanmean(temp)))
            elif outlier_side == 1:
                rm_idx[j] = np.nanargmax(temp)
                sample = np.nanmax(temp)
                R[j] = sample - np.nanmean(temp)
            R[j] = R[j] / np.nanstd(temp)
            temp[int(rm_idx[j])] = np.nan
            p = 1 - alpha / (n - i + 1)
            t = stats.t.ppf(p, n - i - 1)
            lam[j] = ((n - i) * t) / (np.sqrt((n - i - 1 + t**2) * (n - i + 1)))
        mask = np.zeros(n).astype(bool)
        mask[rm_idx[np.where(R > lam)[0]]] = True
        return mask

    # Calculate metric for each window
    metrics = []
    indices = []
    starts = np.arange(0, x.shape[0], window_length)
    for i in range(len(starts)):
        start = starts[i]
        if i == len(starts) - 1:
            stop = None
        else:
            stop = starts[i] + window_length
        m = np.std(x[start:stop])
        metrics.append(m)
        indices += [i] * len(x[start:stop])

    # Detect outliers
    bad_metrics_mask = _gesd(metrics)
    bad_metrics_indices = np.where(bad_metrics_mask)[0]

    # Look up what indices in the original data are bad
    bad_indices = np.isin(indices, bad_metrics_indices)

    return bad_indices

def annotate_bad_segments(raw, picks, **kwargs):
    """Annotate bad segments identified by the G-ESD algorithm."""
    data, times = raw.get_data(picks=picks, reject_by_annotation="omit", return_times=True)
    bad_indices = get_bad_indices(data.T, **kwargs)
    onsets = np.where(np.diff(bad_indices.astype(float)) == 1)[0]
    if bad_indices[0]:
        onsets = np.r_[0, onsets]
    offsets = np.where(np.diff(bad_indices.astype(float)) == -1)[0]
    if bad_indices[-1]:
        offsets = np.r_[offsets, len(bad_indices) - 1]
    onsets_secs = raw.first_samp / raw.info["sfreq"] + times[onsets.astype(int)]
    offsets_secs = raw.first_samp / raw.info["sfreq"] + times[offsets.astype(int)]
    durations_secs = offsets_secs - onsets_secs
    descriptions = np.repeat(f"bad_segment_{picks}", len(onsets))
    raw.annotations.append(onsets_secs, durations_secs, descriptions)
    mod_dur = durations_secs.sum()
    full_dur = raw.n_times / raw.info["sfreq"]
    pc = (mod_dur / full_dur) * 100
    print(f"{picks} bad segments: {mod_dur:.02f}/{full_dur} seconds rejected ({pc:02f}%)")
    return raw

def ica_autoreject(raw, picks, n_components):
    """ICA artefact removal."""    
    # Calculate ICA
    filt_raw = raw.copy().filter(l_freq=1, h_freq=None)
    ica = mne.preprocessing.ICA(n_components=n_components)
    ica.fit(filt_raw, picks=picks)

    # Reject components based on the EOG channel
    eog_indices, eog_scores = ica.find_bads_eog(raw, threshold=0.35, measure="correlation")
    ica.exclude.extend(eog_indices)

    # Reject components based on the ECG channel
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw, threshold="auto", method="ctps")
    ica.exclude.extend(ecg_indices)

    # Remove the components from the data if requested
    ica.apply(raw)

    return raw

def find_events(raw):
    new_event_ids = {"famous": 1, "unfamiliar": 2, "scrambled": 3, "button": 4}
    old_event_ids = {
        "famous": [5, 6, 7],
        "unfamiliar": [13, 14, 15],
        "scrambled": [17, 18, 19],
        "button": [
            256, 261, 262, 263, 269, 270, 271, 273, 274, 275,
            4096, 4101, 4102, 4103, 4109, 4110, 4111, 4114, 4114, 4115,
            4352, 4357, 4359, 4365, 4369,
        ],
    }
    events = mne.find_events(raw, min_duration=0.005, verbose=False)
    for old_event_codes, new_event_codes in zip(old_event_ids.values(), new_event_ids.values()):
        events = mne.merge_events(events, old_event_codes, new_event_codes)
    return events, new_event_ids

def standardize(data):
    data -= np.mean(data, axis=-1, keepdims=True)
    data /= np.std(data, axis=-1, keepdims=True)
    return data

subjects_dir = "../data"
subject = "sub-03"

raw_dir = "../data/raw"
preproc_dir = "../data/preproc"

os.makedirs(preproc_dir, exist_ok=True)

for run in range(1, 7):
    raw_fname = f"{raw_dir}/{subject}/{subject}_ses-meg_task-facerecognition_run-{run:02d}_proc-sss_meg.fif"

    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw = raw.set_channel_types({"EEG061": "eog", "EEG062": "eog", "EEG063": "ecg"})
    raw = raw.filter(l_freq=0.5, h_freq=125)
    raw = raw.notch_filter(freqs=[50, 100])
    raw = raw.resample(sfreq=250)
    raw = annotate_bad_segments(raw, picks="mag", window_length=500)
    raw = annotate_bad_segments(raw, picks="grad", window_length=500)
    raw = ica_autoreject(raw, picks="meg", n_components=40)
    raw.save(f"{preproc_dir}/{subject}_run-{run:02d}_preproc-raw.fif", overwrite=True)

    #raw.apply_function(standardize, picks="meg")
    #events, event_ids = find_events(raw)
    #epochs = mne.Epochs(raw, events, event_ids, tmin=-0.1, tmax=1)
    #epochs.save(f"{preproc_dir}/{subject}_run-{run:02d}_preproc-epo.fif", overwrite=True)
