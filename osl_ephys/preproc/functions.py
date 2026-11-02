"""Preprocessing functions: artefact detection, ICA and headshape decimation.

The detectors mark bad segments, channels and epochs with the generalised
ESD test (:func:`gesd`). The two ICA routes fit on a filtered copy of the
data and remove the artefact components from the original: :func:`ica_label`
labels them with a classifier (mne-icalabel) and
:func:`ica_ecg_eog_correlation` by their correlation with the ECG and EOG
channels. The steps of the ``preproc`` config section call these functions
(:mod:`osl_ephys.preproc.steps`).
"""

from __future__ import annotations

import logging
from os.path import exists

import mne
import numpy as np
from mne.preprocessing import ICA
from scipy import stats

logger = logging.getLogger("osl_ephys")


def gesd(x, alpha=0.05, p_out=0.1, outlier_side=0):
    """Outliers by the generalised ESD test (Rosner, 1983).

    Parameters
    ----------
    x : array_like
        The values to test. NaNs are ignored.
    alpha : float, optional
        Significance level.
    p_out : float, optional
        The largest fraction of the values that can be outliers.
    outlier_side : {-1, 0, 1}, optional
        -1 looks for small outliers, 1 for large ones, 0 for both.

    Returns
    -------
    outliers : np.ndarray of bool
        True where a value is an outlier.
    x : np.ndarray
        The values without the outliers.

    References
    ----------
    B. Rosner (1983). Percentage Points for a Generalized ESD Many-Outlier
    Procedure. Technometrics 25(2), pp. 165-172.
    """
    x = np.asarray(x, dtype=float)
    if np.any(np.isnan(x)):
        finite = np.where(np.isfinite(x))[0]
        outliers_finite, _ = gesd(x[finite], alpha, p_out, outlier_side)
        outliers = np.zeros(x.shape, dtype=bool)
        outliers[finite[outliers_finite]] = True
        return outliers, x[~outliers]
    if outlier_side == 0:
        alpha = alpha / 2
    n = len(x)
    n_out = int(np.ceil(n * p_out))
    temp = x.copy()
    R = np.zeros(n_out)
    removed = np.zeros(n_out, dtype=int)
    lam = np.zeros(n_out)
    for j in range(n_out):
        i = j + 1
        mean = np.nanmean(temp)
        if outlier_side == -1:
            removed[j] = np.nanargmin(temp)
            R[j] = mean - np.nanmin(temp)
        elif outlier_side == 0:
            removed[j] = int(np.nanargmax(abs(temp - mean)))
            R[j] = np.nanmax(abs(temp - mean))
        else:
            removed[j] = np.nanargmax(temp)
            R[j] = np.nanmax(temp) - mean
        R[j] = R[j] / np.nanstd(temp, ddof=1)
        temp[removed[j]] = np.nan
        p = 1 - alpha / (n - i + 1)
        t = stats.t.ppf(p, n - i - 1)
        lam[j] = ((n - i) * t) / (np.sqrt((n - i - 1 + t**2) * (n - i + 1)))
    # The outliers are the first k removed values, for the largest k with R > lam
    significant = np.where(R > lam)[0]
    outliers = np.zeros(n, dtype=bool)
    if significant.size:
        outliers[removed[: significant.max() + 1]] = True
    return outliers, x[~outliers]


def _find_outliers_in_dims(X, axis=-1, metric_func=np.std, gesd_args=None):
    """Outliers along an axis, by a metric computed over the other axes."""
    if axis == -1:
        axis = np.arange(X.ndim)[axis]
    squashed = tuple(np.setdiff1d(np.arange(X.ndim), axis))
    outliers, _ = gesd(metric_func(X, axis=squashed), **(gesd_args or {}))
    return outliers


def _find_outliers_in_segments(
    X,
    axis=-1,
    segment_len=100,
    metric_func=np.std,
    gesd_args=None,
    channel_wise=False,
    channel_axis=0,
    threshold=0.05,
):
    """Outlier segments along an axis: the metric of each non-overlapping segment is
    tested; with ``channel_wise`` each channel is tested on its own and a segment is
    bad when the fraction of bad channels exceeds ``threshold`` (or any channel is
    bad, with ``threshold="any"``)."""
    gesd_args = gesd_args or {}
    if axis == -1:
        axis = np.arange(X.ndim)[axis]
    starts = np.arange(0, X.shape[axis], segment_len)
    n_segments = len(starts)
    segment_of = np.full(X.shape[axis], np.nan)
    slc = [slice(None)] * X.ndim

    def segments():
        for i, start in enumerate(starts):
            stop = None if i == n_segments - 1 else start + segment_len
            slc[axis] = slice(start, stop)
            segment_of[slc[axis]] = i
            yield i

    if channel_wise:
        if channel_axis == -1:
            channel_axis = np.arange(X.ndim)[channel_axis]
        if axis == channel_axis:
            raise ValueError("The time axis and channel axis cannot be the same.")
        n_channels = X.shape[channel_axis]
        if threshold != "any":
            if not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
                raise ValueError("threshold must be between 0 and 1, or 'any'.")
            if n_channels * threshold < 1:
                raise ValueError("threshold * n_channels must be at least 1 channel.")
        metric = np.zeros((n_channels, n_segments))
        for i in segments():
            for ch in range(n_channels):
                slc[channel_axis] = ch
                metric[ch, i] = np.nan_to_num(metric_func(X[tuple(slc)]), nan=0)
            slc[channel_axis] = slice(None)
        bad = np.zeros((n_channels, X.shape[axis]), dtype=bool)
        for ch in range(n_channels):
            outliers, _ = gesd(metric[ch], **gesd_args)
            bad[ch] = np.isin(segment_of, np.where(outliers)[0])
        if threshold == "any":
            return np.any(bad, axis=0)
        return np.sum(bad, axis=0) >= threshold * n_channels

    metric = np.zeros(n_segments)
    for i in segments():
        metric[i] = np.nan_to_num(metric_func(X[tuple(slc)]), nan=0)
    outliers, _ = gesd(metric, **gesd_args)
    return np.isin(segment_of, np.where(outliers)[0])


def detect_artefacts(
    X,
    axis=None,
    reject_mode="dim",
    metric_func=np.std,
    segment_len=100,
    gesd_args=None,
    ret_mode="bad_inds",
    channel_wise=False,
    channel_axis=0,
    channel_threshold=0.05,
):
    """Bad observations or segments of an array.

    Parameters
    ----------
    X : np.ndarray
        The data.
    axis : int
        The axis to look for artefacts along.
    reject_mode : {"dim", "segments"}
        Test each index of the axis (``dim``) or split the axis into
        segments and test those (``segments``).
    metric_func : callable, optional
        The metric tested, a function of an array returning one number.
    segment_len : int, optional
        Length of the segments, in samples.
    gesd_args : dict, optional
        Arguments of :func:`gesd`.
    ret_mode : {"bad_inds", "good_inds", "zero_bads", "nan_bads"}
        What to return: the bad indices, the good indices, or a copy of the
        data with the bad samples zeroed or set to NaN.
    channel_wise : bool, optional
        Test each channel separately (segments only).
    channel_axis : int, optional
        The channel axis, with ``channel_wise``.
    channel_threshold : float or "any", optional
        The fraction of bad channels that makes a segment bad.

    Returns
    -------
    np.ndarray
        A boolean array of length ``X.shape[axis]``, or a copy of the data.
    """
    if reject_mode not in ("dim", "segments"):
        raise ValueError(f"reject_mode '{reject_mode}' not recognised.")
    if ret_mode not in ("bad_inds", "good_inds", "zero_bads", "nan_bads"):
        raise ValueError(f"ret_mode '{ret_mode}' not recognised.")
    if axis is None or axis > X.ndim:
        raise ValueError("bad axis")
    if reject_mode == "dim":
        bad = _find_outliers_in_dims(X, axis, metric_func, gesd_args)
    else:
        bad = _find_outliers_in_segments(
            X,
            axis,
            segment_len,
            metric_func,
            gesd_args,
            channel_wise,
            channel_axis,
            channel_threshold,
        )
    if ret_mode == "bad_inds":
        return bad
    if ret_mode == "good_inds":
        return ~bad
    out = X.copy()
    slc = tuple(bad if i == axis else slice(None) for i in range(X.ndim))
    out[slc] = 0 if ret_mode == "zero_bads" else np.nan
    return out


def detect_maxfilt_zeros(raw, use_maxfilter_log=True):
    """The samples MaxFilter zeroed out, from its log or from the data.

    The log is looked for next to the raw file, with the extension ``.log``.
    Without it, samples where every MEG channel is zero are taken as zeroed.

    Parameters
    ----------
    raw : mne.io.Raw
        The data.
    use_maxfilter_log : bool, optional
        Read the MaxFilter log when it exists.

    Returns
    -------
    bad : np.ndarray of bool
        True at the zeroed samples.
    """
    if not {"mag", "grad"} & set(raw.get_channel_types()):
        logger.info("No MEG data: not looking for MaxFilter zeroed-out data.")
        return np.zeros(raw.n_times, dtype=bool)
    log_file = None
    if raw.filenames[0] is not None:
        log_file = str(raw.filenames[0]).replace(".fif", ".log")
    if use_maxfilter_log and log_file is not None and exists(log_file):
        try:
            with open(log_file) as f:
                lines = f.readlines()
            n_buffers = None
            zeroed = []
            for line in lines:
                if n_buffers is None and " data buffers)" in line:
                    n_buffers = float(line.split("(")[1].split(" data buffers)")[0])
                if ": cont HPI is off, data block is skipped!" in line:
                    zeroed.append(
                        float(
                            line.split("Time ")[1].split(
                                ": cont HPI is off, data block is skipped!"
                            )[0]
                        )
                    )
            buffer_len = raw.n_times / n_buffers
            starts = (np.array(zeroed) - raw.first_time) * raw.info["sfreq"]
            bad = np.zeros(raw.n_times)
            for start in starts:
                bad[int(start) : int(start + buffer_len)] = 1
        except Exception:
            logger.info("Reading the zeroed-out data from the MaxFilter log failed")
            return detect_maxfilt_zeros(raw, use_maxfilter_log=False)
    else:
        logger.info("Detecting zeroed-out data from the data itself")
        bad = np.all(
            raw.get_data(picks="meg", reject_by_annotation="omit") == 0, axis=0
        )
    if np.sum(bad) / len(bad) > 0.5:
        raise RuntimeError(
            "More than half of the data is marked as bad. This often happens when "
            "MaxFilter movement compensation is used but not enough HPI coils are "
            "usable. Check the data and the MaxFilter settings."
        )
    return bad.astype(bool)


def _pick(info, picks, ref_meg="auto", exclude="bads"):
    """The channel indices of a channel type name from a config."""
    kinds = {
        "mag": {"meg": "mag"},
        "grad": {"meg": "grad"},
        "meg": {"meg": True},
        "eeg": {"eeg": True},
        "eog": {"eog": True},
        "ecg": {"ecg": True},
        "emg": {"emg": True},
        "misc": {"misc": True},
    }
    if picks not in kinds:
        raise NotImplementedError(f"picks={picks} not available.")
    kwargs = kinds[picks]
    if picks not in ("eeg", "misc"):
        kwargs["ref_meg"] = ref_meg
    return mne.pick_types(info, exclude=exclude, **kwargs)


def _metric(name):
    if name == "std":
        return np.std
    if name == "var":
        return np.var
    if name == "kurtosis":
        return lambda x: stats.kurtosis(x, axis=None)
    raise ValueError(f"metric must be 'std', 'var' or 'kurtosis', got '{name}'.")


def detect_bad_segments(
    raw,
    picks,
    segment_len=1000,
    significance_level=0.05,
    maximum_fraction=0.1,
    metric="std",
    ref_meg="auto",
    mode=None,
    detect_zeros=True,
    channel_wise=False,
    channel_axis=0,
    channel_threshold=0.05,
):
    """Mark bad segments with the generalised ESD test, as annotations.

    The data of the picked channels is split into non-overlapping segments,
    the metric of each is computed and the outlying segments are annotated
    ``bad_segment_<picks>``. Samples MaxFilter zeroed out are annotated
    ``maxfilter_bad_segment_<picks>``.

    Parameters
    ----------
    raw : mne.io.Raw
        The data.
    picks : str
        Channel type: ``mag``, ``grad``, ``meg``, ``eeg``, ``eog``, ``ecg``,
        ``emg`` or ``misc``. For Elekta data run ``mag`` and ``grad``
        separately. With CTF data ``mag`` with ``ref_meg=False`` gives the
        axial gradiometers and ``grad`` the reference gradiometers.
    segment_len : int, optional
        Length of the segments, in samples.
    significance_level : float, optional
        Significance level of the test.
    maximum_fraction : float, optional
        The largest fraction of the segments that can be marked bad.
    metric : {"std", "var", "kurtosis"}, optional
        The metric tested.
    ref_meg : str, optional
        Passed to :func:`mne.pick_types`.
    mode : {None, "diff", "maxfilter"}, optional
        ``diff`` tests the first difference of the data; ``maxfilter`` only
        marks the zeroed-out samples.
    detect_zeros : bool, optional
        Also mark the samples MaxFilter zeroed out.
    channel_wise : bool, optional
        Test each channel separately; a segment is bad when the fraction of
        bad channels exceeds ``channel_threshold`` (or any channel, with
        ``"any"``).
    channel_axis : int, optional
        The channel axis, with ``channel_wise``.
    channel_threshold : float or "any", optional
        See ``channel_wise``.

    Returns
    -------
    raw : mne.io.Raw
        The data, with the bad segments annotated.
    """
    chs = _pick(raw.info, picks, ref_meg)
    gesd_args = {"alpha": significance_level, "p_out": maximum_fraction}
    zeros = None
    if mode == "maxfilter" or (mode is None and detect_zeros):
        zeros = detect_maxfilt_zeros(raw)
    X, times = raw.get_data(picks=chs, reject_by_annotation="omit", return_times=True)
    if mode == "diff":
        X = np.diff(X, axis=1)
        times = times[1:]
    if mode == "maxfilter":
        found = [(zeros, "maxfilter_")]
    else:
        bad = detect_artefacts(
            X,
            axis=1,
            reject_mode="segments",
            metric_func=_metric(metric),
            segment_len=segment_len,
            gesd_args=gesd_args,
            channel_wise=channel_wise,
            channel_axis=channel_axis,
            channel_threshold=channel_threshold,
        )
        found = [(bad, ""), (zeros, "maxfilter_")]
    for bad, prefix in found:
        if bad is None:
            continue
        onsets = np.where(np.diff(bad.astype(float)) == 1)[0]
        if bad[0]:
            onsets = np.r_[0, onsets]
        offsets = np.where(np.diff(bad.astype(float)) == -1)[0]
        if bad[-1]:
            offsets = np.r_[offsets, len(bad) - 1]
        onsets = raw.first_samp / raw.info["sfreq"] + times[onsets.astype(int)]
        offsets = raw.first_samp / raw.info["sfreq"] + times[offsets.astype(int)]
        durations = offsets - onsets
        raw.annotations.append(
            onsets, durations, [f"{prefix}bad_segment_{picks}"] * len(onsets)
        )
        total = raw.n_times / raw.info["sfreq"]
        logger.info(
            f"{picks}{' (maxfilter)' if prefix else ''}: {len(onsets)} bad segments, "
            f"{durations.sum():.1f}/{total:.1f} seconds ({100 * durations.sum() / total:.1f}%)"
        )
    return raw


def detect_bad_channels(
    raw, picks, ref_meg="auto", significance_level=0.05, log10=True
):
    """Mark bad channels with the generalised ESD test on their standard deviation.

    Parameters
    ----------
    raw : mne.io.Raw
        The data.
    picks : str
        Channel type, as for :func:`detect_bad_segments`.
    ref_meg : str, optional
        Passed to :func:`mne.pick_types`.
    significance_level : float, optional
        Significance level of the test.
    log10 : bool, optional
        Test the log of the standard deviations, which makes them closer to
        normally distributed. A flat channel (zero standard deviation) is
        always bad.

    Returns
    -------
    raw : mne.io.Raw
        The data, with the bad channels added to ``info["bads"]``.
    """
    chs = _pick(raw.info, picks, ref_meg)
    std = np.std(raw.get_data(picks=chs, reject_by_annotation="omit"), axis=-1)
    if log10:
        with np.errstate(divide="ignore"):
            std = np.log10(std)
    finite = np.isfinite(std)
    bad = ~finite
    if np.any(finite):
        bad[finite], _ = gesd(std[finite], alpha=significance_level)
    names = list(np.array(raw.ch_names)[chs][bad])
    logger.info(
        f"{picks}: {len(names)}/{len(chs)} channels rejected "
        f"({100 * len(names) / max(len(chs), 1):.1f}%): {names}"
    )
    raw.info["bads"].extend(n for n in names if n not in raw.info["bads"])
    return raw


def drop_bad_epochs(
    epochs,
    picks,
    significance_level=0.05,
    max_percentage=0.1,
    outlier_side=0,
    metric="std",
    ref_meg="auto",
    mode=None,
):
    """Drop the epochs the generalised ESD test marks as outliers.

    Parameters
    ----------
    epochs : mne.Epochs
        The epochs.
    picks : str
        Channel type, as for :func:`detect_bad_segments`.
    significance_level : float, optional
        Significance level of the test.
    max_percentage : float, optional
        The largest fraction of the epochs that can be dropped.
    outlier_side : {-1, 0, 1}, optional
        See :func:`gesd`.
    metric : {"std", "var", "kurtosis"}, optional
        The metric tested, averaged over channels.
    ref_meg : str, optional
        Passed to :func:`mne.pick_types`.
    mode : {None, "diff"}, optional
        ``diff`` tests the first difference of the data.

    Returns
    -------
    epochs : mne.Epochs
        The epochs, with the bad ones dropped.
    """
    chs = _pick(epochs.info, picks, ref_meg)
    X = epochs.get_data(picks=chs)
    if mode == "diff":
        X = np.diff(X, axis=-1)
    metric_func = stats.kurtosis if metric == "kurtosis" else _metric(metric)
    X = np.mean(metric_func(X, axis=-1), axis=1)
    bad, _ = gesd(X, significance_level, max_percentage, outlier_side)
    logger.info(f"{picks}: {bad.sum()}/{len(X)} epochs rejected")
    epochs.drop(bad)
    return epochs


def detect_bad_channels_psd(raw, fmin=2, fmax=80, n_fft=2000, alpha=0.05):
    """Bad channels by the generalised ESD test on their log power spectrum.

    Parameters
    ----------
    raw : mne.io.Raw
        The data.
    fmin, fmax : float, optional
        Frequency range of the spectrum.
    n_fft : int, optional
        FFT length.
    alpha : float, optional
        Significance level of the test.

    Returns
    -------
    bads : list of str
        The bad channels found (not added to ``info["bads"]``).
    """
    good = [ch for ch in raw.ch_names if ch not in raw.info["bads"]]
    psd = raw.compute_psd(
        fmin=fmin, fmax=fmax, n_fft=n_fft, reject_by_annotation=True, verbose=False
    ).get_data()
    if len(good) != psd.shape[0]:
        raise RuntimeError(f"{len(good)} channels but the PSD has {psd.shape[0]}.")
    broken = [ch for ch, p in zip(good, psd) if np.any(np.isnan(p)) or np.all(p == 0)]
    if broken:
        raise RuntimeError(f"The PSD is NaN or zero for channels: {broken}")
    bad = detect_artefacts(
        np.log10(psd), axis=0, reject_mode="dim", gesd_args={"alpha": alpha}
    )
    return [ch for ch, is_bad in zip(good, bad) if is_bad]


def ica_label(
    raw,
    picks="mag",
    n_components=20,
    method="megnet",
    threshold=0.5,
    random_state=42,
    max_iter="auto",
):
    """ICA with automatic artefact labelling by mne-icalabel.

    The ICA is fitted on a 1-100 Hz copy of the data, the components are
    labelled by a pre-trained classifier (MEGNet for MEG, ICLabel for EEG)
    and the artefact components above the threshold are removed from the
    data.

    Parameters
    ----------
    raw : mne.io.Raw
        The data.
    picks : str, optional
        Channel type for the ICA. MEGNet was trained on magnetometer
        topographies, so ``mag`` for Elekta data; ``eeg`` with
        ``method="iclabel"`` for EEG.
    n_components : int, optional
        Number of components; must not exceed the data rank (about 60 for
        MaxFiltered Elekta data).
    method : {"megnet", "iclabel"}, optional
        The classifier.
    threshold : float, optional
        Probability above which an artefact component is removed.
    random_state : int, optional
        Seed of the ICA.
    max_iter : int or "auto", optional
        Maximum number of ICA iterations.

    Returns
    -------
    raw : mne.io.Raw
        The cleaned data.
    ica : mne.preprocessing.ICA
        The ICA, with ``exclude`` set and the labels in ``labels_``.
    ic_labels : dict
        ``labels`` (list of str) and ``y_pred_proba`` (array).
    """
    from mne_icalabel import label_components

    logger.info(f"ICA labelling: {method}, picks={picks}, {n_components} components")
    raw_fit = raw.copy().filter(l_freq=1.0, h_freq=100.0, verbose=False)
    if method == "iclabel":
        raw_fit.set_eeg_reference("average", verbose=False)
    ica = ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=random_state,
        max_iter=max_iter,
        verbose=False,
    )
    ica.fit(raw_fit, picks=picks)
    logger.info(f"Fitted in {ica.n_iter_}/{ica.max_iter} iterations")
    ic_labels = label_components(raw_fit, ica, method=method)
    labels, probs = ic_labels["labels"], ic_labels["y_pred_proba"]
    # MEGNet returns "brain/other" as one label, ICLabel "brain" and "other"
    exclude = [
        i
        for i, (label, prob) in enumerate(zip(labels, probs))
        if label not in ("brain", "other", "brain/other") and prob > threshold
    ]
    for i in exclude:
        logger.info(f"  ICA{i:03d}: {labels[i]} ({probs[i]:.2f}) excluded")
    ica.labels_ = {ica._ica_names[i]: labels[i] for i in range(ica.n_components_)}
    if exclude:
        ica.exclude = exclude
        ica.apply(raw)
    else:
        logger.info("No artefact components found")
    return raw, ica, ic_labels


def ica_ecg_eog_correlation(
    raw,
    picks="meg",
    n_components=40,
    l_freq=1.0,
    h_freq=None,
    ecg_method="ctps",
    ecg_threshold="auto",
    eog_measure="correlation",
    eog_threshold=0.35,
    random_state=42,
    max_iter="auto",
):
    """ICA with the artefact components found by their ECG and EOG correlation.

    The ICA is fitted on a high-pass filtered copy of the data, the
    components related to the ECG (by cross-trial phase statistics or
    correlation) and to the EOG channels are found with MNE and removed from
    the data.

    Parameters
    ----------
    raw : mne.io.Raw
        The data.
    picks : str, optional
        Channel type for the ICA; ``meg`` fits magnetometers and gradiometers
        together.
    n_components : int, optional
        Number of components; must not exceed the data rank.
    l_freq, h_freq : float, optional
        Filter of the copy the ICA is fitted on.
    ecg_method : {"ctps", "correlation", None}, optional
        ECG detection method; None skips it (needed for EEG without an ECG
        channel, as the synthetic ECG needs magnetometers).
    ecg_threshold : str or float, optional
        Threshold of the ECG detection.
    eog_measure : {"correlation", "zscore"}, optional
        EOG detection measure.
    eog_threshold : float, optional
        Threshold of the EOG detection, a correlation or a z-score.
    random_state : int, optional
        Seed of the ICA.
    max_iter : int or "auto", optional
        Maximum number of ICA iterations.

    Returns
    -------
    raw : mne.io.Raw
        The cleaned data.
    ica : mne.preprocessing.ICA
        The ICA, with ``exclude`` set and the labels in ``labels_``.
    ic_labels : dict
        ``labels`` (list of str) and ``y_pred_proba`` (array), as
        :func:`ica_label` returns them.
    """
    logger.info(f"ICA ECG/EOG correlation: picks={picks}, {n_components} components")
    raw_fit = raw.copy().filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    ica = ICA(
        n_components=n_components,
        method="fastica",
        random_state=random_state,
        max_iter=max_iter,
        verbose=False,
    )
    ica.fit(raw_fit, picks=picks)
    logger.info(f"Fitted in {ica.n_iter_}/{ica.max_iter} iterations")

    ecg_indices, ecg_scores = [], np.zeros(ica.n_components_)
    if ecg_method is not None:
        try:
            ecg_indices, ecg_scores = ica.find_bads_ecg(
                raw_fit, method=ecg_method, threshold=ecg_threshold, verbose=False
            )
        except Exception as e:
            logger.warning(f"ECG detection failed: {e}")
    eog_indices, eog_scores = [], np.zeros(ica.n_components_)
    if len(mne.pick_types(raw_fit.info, eog=True)) == 0:
        logger.info("No EOG channel, skipping EOG detection")
    else:
        try:
            eog_indices, eog_scores = ica.find_bads_eog(
                raw_fit, measure=eog_measure, threshold=eog_threshold, verbose=False
            )
            if isinstance(eog_scores, list):
                eog_scores = np.max(np.abs(eog_scores), axis=0)
        except Exception as e:
            logger.warning(f"EOG detection failed: {e}")
    exclude = sorted(set(ecg_indices) | set(eog_indices))

    labels, probs = [], []
    for i in range(ica.n_components_):
        kinds = [
            k for k, inds in (("ecg", ecg_indices), ("eog", eog_indices)) if i in inds
        ]
        labels.append("+".join(kinds) if kinds else "brain")
        probs.append(
            max(
                abs(ecg_scores[i]) if i in ecg_indices else 0,
                abs(eog_scores[i]) if i in eog_indices else 0,
            )
        )
    ic_labels = {"labels": labels, "y_pred_proba": np.array(probs)}
    ica.labels_ = {"ecg": list(ecg_indices), "eog": list(eog_indices)}
    for i in exclude:
        logger.info(f"  ICA{i:03d}: {labels[i]} ({probs[i]:.2f}) excluded")
    if exclude:
        ica.exclude = exclude
        ica.apply(raw)
    else:
        logger.info("No artefact components found")
    return raw, ica, ic_labels


def decimate_headshape_points(
    raw,
    decimate_amount=0.01,
    include_facial_info=True,
    remove_zlim=-0.02,
    angle=0,
    method="gridaverage",
    face_Z=None,
    face_Y=None,
    face_X=None,
    decimate_facial_info=True,
    decimate_facial_info_amount=0.01,
):
    """Decimate the headshape points of a dense scan (e.g. an EinScan of an OPM
    session) by grid averaging, keeping the fiducials.

    Parameters
    ----------
    raw : mne.io.Raw
        The data, with the headshape in ``info["dig"]``.
    decimate_amount : float, optional
        Grid size in metres.
    include_facial_info : bool, optional
        Keep the points on the face (inside the ``face_*`` ranges).
    remove_zlim : float, optional
        Drop the points below this z (metres), after rotating by ``angle``.
    angle : float, optional
        Rotation about x (degrees) applied before the z cut.
    method : str, optional
        Only ``gridaverage``.
    face_Z, face_Y, face_X : list of float, optional
        Ranges (metres) of the face points.
    decimate_facial_info : bool, optional
        Decimate the face points too.
    decimate_facial_info_amount : float, optional
        Grid size for the face points.

    Returns
    -------
    raw : mne.io.Raw
        The data, with the decimated headshape as its montage.
    """
    dig = raw.info["dig"]
    headshape = np.array([d["r"] for d in dig if "r" in d])
    logger.info(f"Decimating {len(headshape)} digitisation points")
    decimated = _decimate_headshape(
        headshape,
        decimate_amount,
        include_facial_info,
        remove_zlim,
        angle,
        method,
        face_Z if face_Z is not None else [-0.06, 0.02],
        face_Y if face_Y is not None else [0.06, 0.15],
        face_X if face_X is not None else [-0.03, 0.03],
        decimate_facial_info,
        decimate_facial_info_amount,
    )
    fiducials = {"nasion": None, "lpa": None, "rpa": None}
    idents = {2: "nasion", 1: "lpa", 3: "rpa"}
    for point in dig:
        if point["coord_frame"] == 4 and point["ident"] in idents:
            name = idents[point["ident"]]
            if fiducials[name] is None:
                fiducials[name] = point["r"]
    if any(v is None for v in fiducials.values()):
        raise RuntimeError("Nasion, LPA and RPA were not all found in the head frame.")
    montage = mne.channels.make_dig_montage(
        hsp=decimated, coord_frame="head", **fiducials
    )
    logger.info(f"{len(decimated)} points kept")
    return raw.set_montage(montage)


def _decimate_headshape(
    headshape,
    decimate_amount,
    include_facial_info,
    remove_zlim,
    angle,
    method,
    face_Z,
    face_Y,
    face_X,
    decimate_facial_info,
    decimate_facial_info_amount,
):
    if include_facial_info:
        face = (
            (headshape[:, 2] > face_Z[0])
            & (headshape[:, 2] < face_Z[1])
            & (headshape[:, 1] > face_Y[0])
            & (headshape[:, 1] < face_Y[1])
            & (headshape[:, 0] > face_X[0])
            & (headshape[:, 0] < face_X[1])
        )
        facial_points = headshape[face]
        if decimate_facial_info:
            facial_points = _grid_average_decimate(
                facial_points, decimate_facial_info_amount
            )
    if remove_zlim is not None:
        rotated = _rotate_pointcloud(headshape, angle, "x")
        headshape = _rotate_pointcloud(
            rotated[rotated[:, 2] > remove_zlim], -angle, "x"
        )
    if method != "gridaverage":
        raise ValueError(f"Unsupported decimation method: {method}")
    headshape = _grid_average_decimate(headshape, decimate_amount)
    if include_facial_info:
        headshape = np.vstack((headshape, facial_points))
    return headshape


def _rotate_pointcloud(points, angle_degrees, axis="x"):
    """Rotate points about an axis."""
    a = np.radians(angle_degrees)
    c, s = np.cos(a), np.sin(a)
    matrices = {
        "x": [[1, 0, 0], [0, c, -s], [0, s, c]],
        "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
        "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]],
    }
    if axis not in matrices:
        raise ValueError("axis must be 'x', 'y' or 'z'.")
    return np.dot(points, np.array(matrices[axis]).T)


def _grid_average_decimate(points, voxel_size):
    """The mean of the points in each cell of a grid."""
    cells = {}
    for index, point in zip(np.floor(points / voxel_size).astype(np.int32), points):
        cells.setdefault(tuple(index), []).append(point)
    return np.array([np.mean(cell, axis=0) for cell in cells.values()])
