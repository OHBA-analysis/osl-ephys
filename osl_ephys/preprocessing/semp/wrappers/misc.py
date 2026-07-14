import numpy as np
import mne

from osl_ephys.utils.logger import log_or_print
from osl_ephys.preprocessing.semp.utils import proc_userargs


def voltage_correction(dataset, userargs):
    """Rescale EOG/ECG/EMG channels that are stored in uV but marked as volts.

    Some recording systems write auxiliary channels a factor ~1e6 larger than
    the EEG. This detects that (per channel-type RMS vs the median EEG RMS) and
    divides the offenders by 1e6. If nothing exceeds ``ratio_threshold`` the
    stage is a no-op and it says so, so you know it can be dropped from a config
    that does not need it.
    """
    userargs = proc_userargs(userargs, {
        'ratio_threshold': 1000,           # uV-vs-V scale gap
        'picks': ['eog', 'ecg', 'emg'],    # types to check
    })
    ratio_threshold = userargs['ratio_threshold']
    picks = userargs['picks']

    # Reference: median RMS across EEG channels (the "correct" volt scale).
    eeg_data = dataset['raw'].get_data(picks='eeg', units='V')
    ref_rms = np.median(np.sqrt(np.mean(eeg_data**2, axis=1)))

    corrected = []
    for ch_type in picks:
        pick_idx = mne.pick_types(dataset['raw'].info, eeg=False,
                        eog=(ch_type == 'eog'),
                        ecg=(ch_type == 'ecg'),
                        emg=(ch_type == 'emg'),)
        if len(pick_idx) == 0:
            continue

        data = dataset['raw'].get_data(picks=ch_type, units='V')
        ch_rms = np.mean(np.sqrt(np.mean(data**2, axis=1)))

        if (ch_rms / ref_rms) > ratio_threshold:
            log_or_print(
                f"voltage_correction: {ch_type.upper()} RMS ({ch_rms*1e6:.2f} uV) "
                f"is >{ratio_threshold}x the EEG RMS ({ref_rms*1e6:.2f} uV) -- "
                f"rescaling by 1e6 (uV -> V).")
            dataset['raw'].load_data()               # ensure writable, not a memmap
            dataset['raw']._data[pick_idx] /= 1e6
            corrected.append(ch_type)

    if not corrected:
        log_or_print(
            "voltage_correction: no EOG/ECG/EMG channel exceeded the ratio "
            "threshold; scales already look consistent with the EEG. This stage "
            "is a no-op for this dataset and can be removed from the config.")

    return dataset


def cleanup(dataset, userargs):
    userargs = proc_userargs(userargs, {
        'keywords': ['_noise_'],
        'epoch_unload': True,
    })
    keywords = userargs['keywords']
    epoch_unload = userargs['epoch_unload']

    pop_keys = []
    for k in dataset.keys():
        if epoch_unload and '_ep' in k:
            if isinstance(dataset[k], mne.Epochs):
                dataset[k].preload = False
                dataset[k]._data = None

        for keyword in keywords:
            if keyword in k:
                pop_keys.append(k)
                break
    for k in pop_keys:
        dataset.pop(k)
    return dataset


def mid_crop(dataset, userargs):
    """Crops the raw data to the middle of the recording."""
    userargs = proc_userargs(userargs, {
        'length': None,  # Length of the crop in seconds
        'edge': None,    # Edge to leave out from both sides in seconds
    })
    length = userargs['length']
    edge = userargs['edge']

    if length is None and edge is not None:
        tmin = dataset['raw'].times[0] + edge
        tmax = dataset['raw'].times[-1] - edge
    elif length is not None and edge is None:
        tmin = dataset['raw'].times[0]
        tmax = dataset['raw'].times[-1]

        if length > (tmax - tmin):
            raise ValueError(f"Length {length} seconds is longer than the recording duration {tmax - tmin} seconds.")

        mid = (tmin + tmax) / 2
        tmin = mid - length / 2
        tmax = mid + length / 2
    else:
        raise ValueError("Please provide either 'length' or 'edge', not both.")

    dataset['raw'].crop(tmin=tmin, tmax=tmax)
    return dataset
