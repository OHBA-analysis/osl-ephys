"""EEG signal-level helpers the preprocessing wrappers stand on.

- ``pearson_corr``    -- windowed correlation against a template (used by
                         ``correct_trigger``).
- ``correct_trigger`` -- de-jitter periodic triggers by aligning each to one
                         template (``create_epoch`` fixed mode).
- ``mne_epoch2raw``   -- write processed epochs back into a Raw without the
                         epoch-overlap edge artefact (``epoch_aas`` / ``epoch_obs``).
"""
import copy
import numpy as np


def pearson_corr(windows: np.ndarray, template: np.ndarray):
    """
    windows: shape (N, L) — N windows of length L
    template: shape (L,) — single template
    returns: shape (N,) — correlation of each window with the template
    """
    # Normalize template
    return np.array([np.abs(np.corrcoef(window, template)[0,1]) for window in windows])


def correct_trigger(raw, event, event_id, tmin, tmax, template='mid', channel=0, hwin=30):
    """
    Correct all trigger event to **one template** using Pearson correlation.
    The aim is to prevent trigger jitter that can occur due to various factors, mainly asynchronous EEG-fMRI recording.
    Parameters
    ----------
    raw : mne.io.Raw
        The raw data object.
    event : numpy.ndarray, shaped (n_events, 3).
        The event to correct.
    event_id : int
        The event ID to correct.
    tmin : float
        The start time of the epoch relative to the event onset, in seconds.
    tmax : float
        The end time of the epoch relative to the event onset, in seconds.
    template : str, optional
        The template to use for the correction. Can be 'mid', 'start'. Default is 'mid'.
    channel : int, optional
        The channel to use for the correction. Default is 0.
    hwin : int, optional
        The half window size for the best trigger searching. Default is 30 (timepoints).
    Returns
    -------
    corrected_events : np.ndarray
        The corrected events array. Only contains events with the specified event_id.
    """

    event = event[event[:, 2] == event_id, 0]
    new_events = []
    sfreq = raw.info['sfreq']
    data = raw.get_data()[channel]
    tpmin = int(tmin * sfreq)
    tpmax = int(tmax * sfreq)

    if template == 'mid':
        tmplt_event_tp = event[event.shape[0] // 2] - raw.first_samp
        tmplt = data[tmplt_event_tp + tpmin:tmplt_event_tp + tpmax+1]
    elif template == 'start':
        try:
            tmplt_event_tp = event[0] - raw.first_samp
            tmplt = data[tmplt_event_tp + tpmin:tmplt_event_tp + tpmax+1]
        except IndexError:
            tmplt_event_tp = event[1] - raw.first_samp
            tmplt = data[tmplt_event_tp + tpmin:tmplt_event_tp + tpmax+1]
    else:
        raise ValueError(f"Template {template} not supported. Use 'mid' or 'start'.")
    best_pos_list = []

    for ev_tp in event:
        ev_tp -= raw.first_samp
        win_pos_list = np.arange(ev_tp-hwin, ev_tp+hwin+1)
        win_pos_list = win_pos_list[(win_pos_list+tpmin >= 0) & (win_pos_list+tpmax+1 < len(data))]
        if len(win_pos_list) == 0:
            continue

        window_arr = np.stack([data[pos+tpmin:pos+tpmax+1] for pos in win_pos_list])
        corr = pearson_corr(window_arr, tmplt)
        best_pos = win_pos_list[np.argmax(corr)]
        best_pos_list.append(best_pos-ev_tp)

        new_events.append([best_pos + raw.first_samp, 0, event_id])

    return np.array(new_events, dtype=np.int32)


def mne_epoch2raw(epoch, raw, ndarray=None, tmin=0, overwrite='new', picks='eeg'):
    """ Convert an mne.Epochs object to an mne.RawArray object by overwriting the raw data.
    An easier way to realize this is to 1) use mne.Epochs.get_data() to get the data in ndarray format, 2) reshape, and then 3) use mne.io.RawArray to create a new Raw object. However, this approach is dangerous if the epoch data have overlapping time windows, which is common in BCG correction, or FASTR GA correction. In this case artifact would appear at the edge of concatenated epochs.
    Instead, this function directly overwrites the data in the raw object, which can avoid the edge artifact issue. The overwrite parameter specifies the behavior when overwriting the raw data. 'new' means the epoch with a larger index will overwrite the epoch with a smaller index, while 'even' means the datapoint closer to the event onset will be retained.
    Parameters
    ----------
    epoch : mne.Epochs
        The mne.Epochs object containing the epoched data to be converted. If ndarray is not None, the data from this object will not be used, only metadata such as events and channel names will be used.
    raw : mne.io.Raw
        The mne.Raw object to be overwritten with the epoched data.
    ndarray : numpy.ndarray, optional
        A numpy array containing the data to be used for conversion. If None, the data from the epoch object will be used.
    tmin : float, optional
        The start time of the epoch relative to the event onset, in seconds. Default is 0.
    overwrite : str, optional
        Specifies the behavior when overwriting the raw data. Default is 'new'.
        'new' : The epoch with a larger index will overwrite the epoch with a smaller index.
        'even' : The datapoint closer to the event onset will be retained.
        'obs' : early dirty epochs. for first 11 epochs, use new, for the rest, use even. This is following the OBS method in Niazy05.
    picks : str, optional
        The channels to include in the conversion. Default is 'eeg'. Would raise an AssertionError if the number of channels in the ndarray object does not match the number of channels in "picks" in the raw object.
    Returns
    -------
    raw : mne.io.Raw
        The mne.Raw object with the epoched data overwritten.
    Raises
    ------
    AssertionError
        If the number of channels in the ndarray object does not match the number of channels in "picks" in the raw object.
    """

    raw = copy.deepcopy(raw)
    epoch = copy.deepcopy(epoch).pick(picks)
    picked_idx = [raw.ch_names.index(ch) for ch in epoch.ch_names]

    if len(raw.info['bads']) > 0:
        picked_idx = [idx for idx in picked_idx if raw.ch_names[idx] not in raw.info['bads']]

    # get data
    processed_data = ndarray if ndarray is not None else epoch.get_data()

    # check if the number of channels in the ndarray object matches the number of channels in "picks" in the raw object
    assert processed_data.shape[1] == len(picked_idx), f"Number of channels in ndarray ({processed_data.shape[1]}) does not match number of channels in raw object ({len(picked_idx)})"

    sfreq = raw.info['sfreq']
    tmin_shift = sfreq*tmin

    old_mid = -100000
    for i, event_onset in enumerate(epoch.events[:,0]):
        start_sample = int(event_onset+tmin_shift) - raw.first_samp
        end_sample = start_sample+processed_data.shape[2]
        epoch_data = processed_data[i]
        if overwrite == 'even' or (overwrite == 'obs' and i > 10):
            mid_sample = start_sample + processed_data.shape[2]//2 + 1
            epoch_divide = (mid_sample + old_mid) // 2 + 1
            if epoch_divide > start_sample:
                epoch_data = epoch_data[:, epoch_divide-start_sample:]
                start_sample = epoch_divide
            old_mid = mid_sample

        if end_sample > raw._data.shape[1] or start_sample < 0:
            raise ValueError(f"epoch {i} exceeds raw data bound {raw._data.shape[1]}. ({start_sample}~{end_sample})")

        raw._data[picked_idx, start_sample:end_sample] = epoch_data

    return raw
