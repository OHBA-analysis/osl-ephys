import copy

import numpy as np
import mne
from osl_ephys.utils.logger import log_or_print

from ..utils import proc_userargs, correct_trigger, require_keys


def _first_present(ev_id, event_key):
    """First candidate label present in an ``events_from_annotations`` id map.

    ``event_key`` may be a single label or a list of candidates; returns
    ``(event_id, label)`` for the first one found (robust to the label drifting
    between sessions/sites). Raises if none are present.
    """
    candidates = [event_key] if isinstance(event_key, (str, int)) else list(event_key)
    for name in candidates:
        if str(name) in ev_id:
            return ev_id[str(name)], str(name)
    raise ValueError(
        f"None of the event names {[str(c) for c in candidates]} are in the raw "
        f"annotations {list(ev_id.keys())}. Check the event key or the annotations."
    )


def _resolve_event(raw, event_key):
    """``_first_present`` against ``raw``'s annotations (parses them once)."""
    return _first_present(mne.events_from_annotations(raw)[1], event_key)


def _events_from_timepoints(timepoints):
    """(N, 3) MNE events array (all code 1) from a 1-D list of sample onsets."""
    col = np.asarray(timepoints, dtype=np.int64).reshape(-1, 1)
    return np.concatenate([col, np.zeros_like(col), np.ones_like(col)], axis=1)


def _duration_to_n_samples(duration, sfreq, name):
    """Convert a positive duration to its nearest integer sample count."""
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"{name} must be a positive finite duration in seconds.")
    n_samples = int(round(duration * sfreq))
    if n_samples < 1:
        raise ValueError(f"{name} is shorter than one sample at {sfreq:g} Hz.")
    return n_samples


def _deduplicate_event_samples(events, event_id, name):
    """Select one event code and retain one row per absolute sample.

    Some EEGLAB files contain the same annotation more than once at one sample.
    MNE refuses to construct epochs from repeated event samples.  Selecting the
    requested code first prevents unrelated simultaneous annotations from
    interfering; stable sample sorting and first-row retention then remove only
    exact duplicates, never merely close events.
    """
    selected = np.asarray(events)[np.asarray(events)[:, -1] == event_id]
    if not len(selected):
        raise ValueError(f"{name}: no events remain for event id {event_id!r}.")

    order = np.argsort(selected[:, 0], kind="stable")
    selected = selected[order]
    keep = np.r_[True, np.diff(selected[:, 0]) != 0]
    n_duplicates = int((~keep).sum())
    if n_duplicates:
        log_or_print(
            f"{name}: removed {n_duplicates} duplicate event(s) occurring at "
            "an already-used sample."
        )
    return selected[keep]


def _check_fixed_spacing(onsets, sfreq, expected_interval=None, first_samp=0,
                         jitter_tol=0.1, name='create_epoch', max_report=6):
    """Sanity-check fixed-mode trigger spacing; warn on likely forgotten /
    spurious events.

    Heuristic (on the sorted trigger onsets):

    1. ``d = np.diff(onsets)``; compare each gap to the **median** gap ``m``
       (median is robust to a handful of anomalies).
    2. gaps within ``+/- jitter_tol`` of ``m`` are "on grid" -- small jitter is
       *not* reported.
    3. a gap ``~= k*m`` (integer ``k>=2``) is a **forgotten** trigger: ``k-1``
       missing in between.
    4. a gap ``< 0.6*m`` is a **spurious / duplicated** trigger.
    5. anything else, on an otherwise-regular grid, is flagged as irregular.

    If the gaps are *not* clustered around one value at all (fewer than half on
    grid), it instead warns that the spacing is neither fixed nor near-constant
    -- usually the wrong ``event_key`` / ``mode``. With ``expected_interval``
    (seconds; ``create_TR_epoch`` passes the TR) a median that disagrees with it
    is also flagged (wrong ``tr_interval`` / trigger).

    Returns a findings dict; also emits warnings via ``log_or_print``. Purely
    diagnostic -- never mutates events or raises.
    """
    onsets = np.sort(np.asarray(onsets, dtype=np.float64))
    diffs = np.diff(onsets)
    if diffs.size < 2:
        return {'status': 'too_few'}
    med = float(np.median(diffs))
    if med <= 0:
        log_or_print(f"{name}: cannot check trigger spacing (non-positive median gap).")
        return {'status': 'degenerate'}

    rel = diffs / med
    on_grid = np.abs(rel - 1.0) <= jitter_tol
    k = np.round(rel)
    is_gap = (k >= (2*(1-jitter_tol))) & (np.abs(rel - k) <= 0.25)              # forgotten
    is_short = (rel < 0.6) & ~is_gap                            # spurious/dup
    is_irregular = ~on_grid & ~is_gap & ~is_short              # off-grid, unclassified
    frac_on = float(on_grid.mean())

    def _where(mask, tag=None):
        idx = np.nonzero(mask)[0]
        parts = []
        for i in idx[:max_report]:
            t = (onsets[i] - first_samp) / sfreq
            parts.append(f"~{t:.1f}s(x{int(k[i])})" if tag == 'gap' else f"~{t:.1f}s")
        more = "" if idx.size <= max_report else f", +{idx.size - max_report} more"
        return ", ".join(parts) + more

    findings = {'status': 'grid', 'median_s': med / sfreq, 'frac_on_grid': frac_on,
                'n_missing': int(np.sum(k[is_gap] - 1)) if is_gap.any() else 0,
                'n_short': int(is_short.sum()), 'n_irregular': int(is_irregular.sum())}

    if frac_on < 0.5:
        findings['status'] = 'irregular'
        log_or_print(
            f"{name}: WARNING -- trigger spacing is neither fixed nor clustered "
            f"around one value (only {frac_on:.0%} of {diffs.size} intervals are "
            f"near the median {med/sfreq:.3f}s). Are you sure event_key/mode are "
            f"right for a fixed-window epoch?")
        return findings

    if is_gap.any():
        log_or_print(
            f"{name}: WARNING -- {findings['n_missing']} trigger(s) look "
            f"FORGOTTEN: {int(is_gap.sum())} gap(s) ~= an integer multiple of the "
            f"median {med/sfreq:.3f}s at {_where(is_gap, 'gap')}. A missed trigger "
            f"leaves a stretch with no epoch -- add the missing trigger(s) in your "
            f"initialize (raw.annotations.append(onset, 0, <event_key>)).")
    if is_short.any():
        log_or_print(
            f"{name}: WARNING -- {int(is_short.sum())} SHORT interval(s) "
            f"(< 0.6x the median {med/sfreq:.3f}s) at {_where(is_short)}: likely a "
            f"spurious or duplicated trigger.")
    if is_irregular.any():
        log_or_print(
            f"{name}: note -- {int(is_irregular.sum())} interval(s) off-grid but "
            f"neither a clean multiple nor a short spike at {_where(is_irregular)}.")

    if expected_interval is not None and abs(med / sfreq - expected_interval) / expected_interval > jitter_tol:
        log_or_print(
            f"{name}: WARNING -- median trigger spacing {med/sfreq:.3f}s differs "
            f"from the expected window {expected_interval:.3f}s by >{jitter_tol:.0%}; "
            f"wrong tr_interval/tmax or event_key?")

    if not (is_gap.any() or is_short.any() or is_irregular.any()):
        findings['status'] = 'ok'
    return findings


def crop_TR(dataset, userargs):
    """
    Crops the dataset to the TRs of the fMRI data.
    userargs{event_reference: bool} - If True, after cropping, the event would be overwritten to the event in dataset["raw"].

    Create the TR epochs before this stage when possible.  If no epoch source
    is present, a warning is emitted because cropping first can leave the
    trigger-correction search asymmetric at the new raw boundary.  Cropping
    can also drop a trigger whose complete TR window exceeds that boundary.
    By default cached epochs are invalidated because their raw coordinates are
    no longer guaranteed to match.  For an intentional
    ``create_TR_epoch -> crop_TR`` sequence, pass ``preserve_epochs=True``:
    epoch windows are checked against the cropped raw, invalid windows are
    dropped with a warning, and valid epochs retain their absolute samples.

    ``TR`` defaults to ``dataset['tr_interval']`` and ``event_name`` to
    ``dataset['tr_event_key']`` (both set by the project ``initialize``).
    """
    userargs = proc_userargs(userargs, {
        'TR': dataset.get('tr_interval'),
        'tmin': 0.0,
        'event_name': dataset.get('tr_event_key'),
        'num_edge_TR': 0,
        'preserve_epochs': False,
    })
    TR = userargs['TR']
    tmin = userargs['tmin']
    event_name = userargs['event_name']
    num_edge_TR = userargs['num_edge_TR']
    preserve_epochs = userargs['preserve_epochs']

    if TR is None:
        raise ValueError("crop_TR needs 'TR' (or dataset['tr_interval']).")
    if event_name is None:
        raise ValueError("crop_TR needs 'event_name' (or dataset['tr_event_key']).")

    cached_epoch_keys = [
        key for key, value in dataset.items()
        if key != "raw" and isinstance(value, mne.BaseEpochs)
    ]
    if not cached_epoch_keys:
        log_or_print(
            "Warning: crop_TR is running before TR epoch creation. Create "
            "the TR epochs before cropping so trigger correction can use the "
            "full recording and the AAS epoch source remains aligned."
        )

    freq = dataset['raw'].info['sfreq']
    _, event_name = _resolve_event(dataset['raw'], event_name)   # -> single label

    def crop_eeg_to_tr(eeg, tmin, num_edge_TR=0):
        events_arr, ev_id = mne.events_from_annotations(eeg)
        trig = ev_id[str(event_name)]

        # Collect every TR trigger's onset (samples, recording-relative).
        tr_samples = [tp - eeg.first_samp for tp, _, tv in events_arr if tv == trig]
        if not tr_samples:
            raise ValueError(f"No TR events ({event_name}) found in raw annotations.")

        n_samples = eeg.n_times  # recording length in samples
        tr_in_samples = _duration_to_n_samples(TR, freq, "crop_TR TR")

        # Drop TR onsets that fall outside the recording window:
        #   * onset < 0                       -> annotation lies before data start
        #     (e.g. trigger from a prior segment carried over, or first_samp drift)
        #   * onset + TR*sfreq > n_samples    -> [onset, onset + TR) overruns end
        #     (last TR truncated, TR value wrong, or stray late annotation)
        n_under = sum(1 for s in tr_samples if s < 0)
        n_over = sum(1 for s in tr_samples if s + tr_in_samples > n_samples)
        kept = [s for s in tr_samples if 0 <= s and s + tr_in_samples <= n_samples]
        n_dropped = len(tr_samples) - len(kept)
        if n_dropped:
            log_or_print(
                f"Warning: dropped {n_dropped}/{len(tr_samples)} TR event(s) "
                f"({n_under} before data start, {n_over} past end-of-recording "
                f"[{n_samples / freq:.3f}s] with TR={TR}s). "
                f"Possible causes: last TR truncated, TR value wrong, stray "
                f"annotation outside data window, or first_samp drift."
            )
            if any(isinstance(value, mne.BaseEpochs)
                   for key, value in dataset.items() if key != "raw"):
                cache_action = (
                    "will be checked and filtered when preserve_epochs=True"
                    if preserve_epochs else
                    "will be invalidated"
                )
                log_or_print(
                    "crop_TR: an existing epoch set may now be missing the "
                    "dropped TR event(s); cached epochs {}. Recreate epochs "
                    "after checking the boundary if needed.".format(cache_action)
                )
            # Strip the offending annotations so downstream stages don't see them.
            def _is_outside(onset_s, desc):
                if str(desc) != str(event_name):
                    return False
                s = (onset_s - eeg.first_time) * freq
                return s < 0 or s + tr_in_samples > n_samples

            keep_mask = [
                not _is_outside(o, d)
                for o, d in zip(eeg.annotations.onset, eeg.annotations.description)
            ]
            eeg.set_annotations(eeg.annotations[keep_mask])
            if not kept:
                raise ValueError(
                    "All TR events fell outside the data window; nothing left to crop to."
                )

        start_point = kept[0]
        # ``end_point`` is the final included sample, not the first sample
        # after the final TR.  Each retained TR therefore spans exactly
        # ``tr_in_samples`` samples: [onset, onset + tr_in_samples - 1].
        end_point = kept[-1] + tr_in_samples - 1

        new_tmin = max(start_point / freq + tmin + num_edge_TR * TR, 0.0)
        tmax = min(end_point / freq - num_edge_TR * TR, eeg.times[-1])
        # ``tmax`` names the exact final sample we want, so retain it.
        eeg = eeg.crop(tmin=new_tmin, tmax=tmax, include_tmax=True)
        return eeg

    dataset["raw"] = crop_eeg_to_tr(dataset["raw"], tmin=tmin, num_edge_TR=num_edge_TR)

    # Cropping changes the raw sample window (including ``first_samp``).  The
    # default is to invalidate cached epochs/events.  An intentional
    # create-before-crop pipeline may preserve preloaded epochs whose absolute
    # event windows still fit the cropped raw.
    invalidated = []
    preserved = []
    for key, value in list(dataset.items()):
        if key == "raw":
            continue
        if isinstance(value, mne.BaseEpochs):
            if not preserve_epochs:
                invalidated.append(key)
                del dataset[key]
                continue

            epoch_sfreq = value.info['sfreq']
            if not np.isclose(epoch_sfreq, dataset['raw'].info['sfreq']):
                invalidated.append(key)
                del dataset[key]
                log_or_print(
                    "crop_TR: cannot preserve {} because its sampling rate "
                    "differs from the cropped raw; recreate it.".format(key)
                )
                continue

            starts = (
                value.events[:, 0]
                + int(round(value.tmin * epoch_sfreq))
                - dataset['raw'].first_samp
            )
            # MNE Epochs exposes the epoch length through ``times`` rather
            # than a Raw-like ``n_times`` attribute.
            ends = starts + len(value.times)
            keep = (starts >= 0) & (ends <= dataset['raw'].n_times)
            n_drop = int((~keep).sum())
            if n_drop:
                log_or_print(
                    "crop_TR: dropping {} cached epoch(s) from {} because "
                    "their complete windows do not fit the cropped raw.".format(
                        n_drop, key
                    )
                )
                value = value[keep]
                dataset[key] = value
            if len(value):
                preserved.append(key)
            else:
                invalidated.append(key)
                del dataset[key]
    if isinstance(dataset.get("events"), np.ndarray):
        dataset["events"] = None
        dataset["event_id"] = None
        invalidated.append("events")
    if invalidated:
        log_or_print(
            "crop_TR: raw window changed; invalidated cached "
            + ", ".join(invalidated)
            + ". Recreate epochs from the cropped raw."
        )
    if preserved:
        log_or_print(
            "crop_TR: preserved validated cached epoch(s): "
            + ", ".join(preserved)
            + ". Their event samples remain absolute."
        )
    return dataset


def crop_by_epoch(dataset, userargs):
    """Crop raw to the complete sample span covered by cached epochs.

    Epoch event samples are absolute MNE sample numbers.  Deriving the crop
    from ``len(epoch.times)`` avoids the inclusive-``tmax`` off-by-one that can
    otherwise remove the last sample of the final epoch.
    """
    userargs = proc_userargs(userargs, {
        'epoch_key': 'sim_ep',
        'num_edge_epoch': 0,
    })
    epoch_key = userargs['epoch_key']
    num_edge_epoch = userargs['num_edge_epoch']

    epoch = dataset[epoch_key]
    events = copy.deepcopy(epoch.events)
    events = events[np.argsort(events[:, 0])]  # sort events by timepoint
    if not len(events):
        raise ValueError(f"{epoch_key!r} contains no epochs to crop around.")

    raw = dataset['raw']
    sfreq = raw.info['sfreq']
    epoch_start_shift = int(round(epoch.tmin * sfreq))
    first_sample = events[0, 0] + epoch_start_shift
    last_sample = events[-1, 0] + epoch_start_shift + len(epoch.times) - 1

    edge_samples = num_edge_epoch * len(epoch.times)
    first_sample += edge_samples
    last_sample -= edge_samples
    if first_sample > last_sample:
        raise ValueError(
            "num_edge_epoch removes the complete epoch-covered data span."
        )

    first_sample = max(first_sample, raw.first_samp)
    last_sample = min(last_sample, raw.last_samp)
    tmin = (first_sample - raw.first_samp) / sfreq
    tmax = (last_sample - raw.first_samp) / sfreq
    dataset["raw"] = raw.crop(tmin=tmin, tmax=tmax, include_tmax=True)
    return dataset


def create_epoch(dataset, userargs):
    """Cut epochs locked to a periodic trigger (or a synthetic grid) and stash
    them in the dataset for the artefact-template stages (``epoch_aas`` /
    ``epoch_obs``, which read them by key).

    ``mode`` selects how the epoch **window** is set (this replaces the old
    ``event='TR'/'He'`` switch):

    - ``'fixed'`` (default) -- you provide ``tmin`` and ``tmax``; every epoch
      spans that same window. Use it when the period is a known constant.
      **This is the mode you want for TR (fMRI-volume) epochs.**
    - ``'auto'`` -- the window is estimated from the data: ``tmin`` defaults to
      0 and ``tmax`` is the median inter-event interval (x1.02, capped at the
      largest gap). Use it when the period is not known a priori, or when the 
      trigger is not perfectly periodic.
      **This is the mode you want for helium-pump (He) epochs.**

    For a *triggerless* fixed grid (no event to lock onto -- e.g. controlled
    tests on ex-scanner EEG), use the separate :func:`simulate_epoch` wrapper.
    It is kept apart on purpose: there ``jitter`` perturbs a regular grid, which
    is a different idea from ``create_epoch``'s ``random`` (below), so the two do
    not share one overloaded name.

    ``event_key`` is the annotation label (or list of candidate labels) to epoch
    on (required). ``correct_trig`` (fixed mode) Pearson-aligns the triggers
    first; ``random`` replaces the real triggers with **surrogate** epochs at
    random onsets over the same span.

    For the common cases prefer the thin convenience wrappers
    :func:`create_TR_epoch` (fixed, one TR from ``dataset['tr_interval']``) and
    :func:`create_He_epoch` (auto, from ``dataset['he_event_key']``); reach for
    ``create_epoch`` directly only when you need a non-default window or mode.
    """
    if 'event' in userargs:
        raise KeyError(
            "create_epoch no longer takes 'event' (the old 'TR'/'He' switch). "
            "Use mode='fixed'|'auto' with an explicit event_key (or the "
            "create_TR_epoch / create_He_epoch wrappers); for a triggerless "
            "grid use simulate_epoch.")

    userargs = proc_userargs(userargs, {
        'mode': 'fixed',
        'event_key': None,
        'tmin': 0.0,
        'tmax': None,
        'random': False,
        'correct_trig': False,
        'epoch_key': None,
        'check_spacing': True,   # fixed mode: warn on forgotten/spurious triggers
        'expected_interval': None,
    })
    
    mode = userargs['mode']
    if mode not in ('fixed', 'auto'):
        raise ValueError(
            f"mode {mode!r} not recognized; use 'fixed' or 'auto' "
            f"(for a triggerless grid use simulate_epoch).")

    event_key = userargs['event_key']
    tmin = userargs['tmin']
    tmax = userargs['tmax']
    random = userargs['random']
    correct_trig = userargs['correct_trig']
    epoch_key = userargs['epoch_key']
    check_spacing = userargs['check_spacing']
    expected_interval = userargs['expected_interval']
    sfreq = dataset['raw'].info['sfreq']

    if event_key is None:
        raise ValueError(
            f"mode={mode!r} needs 'event_key' (annotation label[s] to epoch on).")
    if epoch_key is None:
        raise ValueError(
            f"mode={mode!r} needs 'epoch_key' (dataset key to store the epochs).")

    # Fast path: reuse an existing epoch set, only rescaling the sample indices
    # if the raw was resampled since (keeps onsets aligned to the new sfreq).
    # Otherwise resolve the trigger from the annotations.
    if isinstance(dataset.get(epoch_key), mne.Epochs):
        cached = dataset[epoch_key]
        events = cached.events.copy()
        events[:, 0] //= int(cached.info['sfreq'] // sfreq)
        event_id = next(iter(cached.event_id.values()))
    else:
        all_events, ev_id = mne.events_from_annotations(dataset['raw'])
        event_id, _ = _first_present(ev_id, event_key)
        events = all_events
        if mode == 'fixed' and correct_trig:
            events = correct_trigger(dataset['raw'], events, event_id,
                                     tmin=tmin, tmax=tmax, template='mid',
                                     channel=0, hwin=3)

    # Work only with the requested event code and remove exact repeated sample
    # annotations before interval diagnostics or MNE Epochs construction.
    events = _deduplicate_event_samples(events, event_id, epoch_key)

    if mode == 'auto':
        # size the window from the trigger spacing (period unknown a priori)
        gaps = np.diff(events[:, 0])
        tmax = min(np.median(gaps) * 1.02, np.max(gaps)) / sfreq
    elif tmax is None:
        raise ValueError("mode='fixed' needs 'tmin' and 'tmax' (the epoch window in s).")

    # fixed mode assumes a periodic trigger (window ~= period): warn if the
    # spacing looks like it has forgotten / spurious events (diagnostic only).
    if mode == 'fixed' and check_spacing and not random:
        if expected_interval is None:
            expected_interval = tmax - tmin
        _check_fixed_spacing(
            events[:, 0], sfreq, expected_interval=expected_interval,
            first_samp=int(dataset['raw'].first_samp), name=epoch_key)

    if random:
        tps = events[:, 0]
        surrogate = np.sort(np.random.choice(
            np.arange(tps.min(), tps.max()), size=len(tps), replace=False))
        events, event_id = _events_from_timepoints(surrogate), 1

    # if random and epoch_key.endswith('_ep'):
    #     epoch_key = epoch_key + '_rand'   # tr_ep -> tr_ep_rand, he_ep -> he_ep_rand

    dataset[epoch_key] = mne.Epochs(
        dataset['raw'], events=events, tmin=tmin, tmax=tmax, event_id=event_id,
        baseline=None, proj=False, preload=True)
    return dataset


def create_TR_epoch(dataset, userargs):
    """TR-locked epochs, the easy way: one epoch per fMRI-volume trigger, window
    = one TR. A thin, less-customizable :func:`create_epoch` (mode='fixed') that
    reads ``dataset['tr_interval']`` (epoch duration) and ``dataset['tr_event_key']``
    (trigger label), so in the common case a config only needs
    ``{'create_TR_epoch': {}}``. A duration of ``TR`` seconds at sampling rate
    ``sfreq`` always produces exactly ``round(TR * sfreq)`` samples; adjacent
    regularly spaced TR epochs therefore do not share a boundary sample.
    Override any of ``tmin`` / ``duration`` /
    ``event_key`` / ``epoch_key`` / ``correct_trig`` / ``random`` /
    ``check_spacing`` as needed, or drop to ``create_epoch`` for a non-TR window.

    ``check_spacing`` (default True) warns when the volume triggers look like
    they have forgotten or spurious events -- e.g. one gap ~2x the rest usually
    means a missed trigger (add it in your ``initialize``/``on_init``).
    """
    userargs = proc_userargs(userargs, {
        'event_key': dataset.get('tr_event_key'),
        'tmin': 0.0,
        'duration': dataset.get('tr_interval'),
        'epoch_key': 'tr_ep',
        'correct_trig': True,
        'random': False,
        'check_spacing': True,
    })
    duration = userargs.pop('duration')
    if duration is None:
        raise KeyError(
            "create_TR_epoch needs dataset['tr_interval'] (or an explicit duration)."
        )
    sfreq = dataset['raw'].info['sfreq']
    n_samples = _duration_to_n_samples(duration, sfreq, "create_TR_epoch duration")
    userargs['tmax'] = userargs['tmin'] + (n_samples - 1) / sfreq
    userargs['expected_interval'] = float(duration)
    return create_epoch(dataset, {'mode': 'fixed', **userargs})


def create_He_epoch(dataset, userargs):
    """Helium-pump-locked epochs, the easy way: one epoch per He trigger, window
    auto-sized from the trigger spacing. A thin :func:`create_epoch`
    (mode='auto') that reads ``dataset['he_event_key']``; in the common case a
    config only needs ``{'create_He_epoch': {}}``.
    """
    userargs = proc_userargs(userargs, {
        'event_key': dataset.get('he_event_key'),
        'tmin': 0.0,
        'epoch_key': 'he_ep',
        'random': False,
    })
    return create_epoch(dataset, {'mode': 'auto', **userargs})


def simulate_epoch(dataset, userargs):
    """Lay epochs on a fixed grid with **no trigger** to lock onto.

    For controlled tests on ex-scanner ("clean-room") EEG where you want
    TR-like epochs but the recording carries no volume trigger. Deliberately a
    separate wrapper from :func:`create_epoch`: its ``jitter`` knob perturbs a
    regular grid, which is a different idea from ``create_epoch``'s ``random``
    (which *replaces* real triggers with surrogate onsets) -- so one name never
    means two things.

    userargs:
      * ``width``      -- epoch span == grid spacing, in seconds (required).
      * ``jitter``     -- fraction of ``width`` to randomly shift each onset by
                          (default 0.0 == a perfectly regular grid).
      * ``tmin``       -- epoch start relative to each grid point (default 0.0).
      * ``epoch_key`` -- output key (default ``'sim_ep'``).
    """
    userargs = proc_userargs(userargs, {
        'width': None,
        'jitter': 0.0,
        'tmin': 0.0,
        'epoch_key': 'sim_ep',
    })
    width = userargs['width']
    if width is None:
        raise ValueError("simulate_epoch needs 'width' (the epoch/grid width in s).")
    jitter_frac = userargs['jitter']
    tmin = userargs['tmin']
    epoch_key = userargs['epoch_key']

    sfreq = dataset['raw'].info['sfreq']
    step = _duration_to_n_samples(width, sfreq, "simulate_epoch width")
    tps = np.arange(
        dataset['raw'].first_samp,
        dataset['raw'].last_samp + 1,
        step,
        dtype=np.int64,
    )
    jitter = int(step * jitter_frac)
    if jitter > 0:
        tps = tps + np.random.randint(-jitter, jitter, size=len(tps))

    # MNE includes both tmin and tmax.  A tmax equal to width would therefore
    # add one shared boundary sample to adjacent epochs.  End one sample early
    # so jitter=0 produces exact, non-overlapping [onset, onset + width) spans.
    tmax = tmin + (step - 1) / sfreq
    dataset[epoch_key] = mne.Epochs(
        dataset['raw'], events=_events_from_timepoints(tps), tmin=tmin,
        tmax=tmax, event_id=1, baseline=None, proj=False, preload=True)
    return dataset
