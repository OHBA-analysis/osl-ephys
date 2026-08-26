"""Plotting helpers used by the SEMP preprocessing tutorial and reports.

The public functions in this module deliberately keep plotting code explicit:

* :func:`psd_plot` prepares a finite, contiguous copy of the signal before
  estimating a PSD.  This is important after bad-segment detection, because
  MNE represents rejected spans as ``NaN`` samples.
* :func:`temp_plot` draws one channel and, optionally, event markers.
* :func:`temp_plot_diff` compares two recordings after aligning their sample
  ranges.
* :func:`pcs_plot` writes time-domain and PSD panels for PCA components.

The helpers below each implement one small plotting step.  Keeping input
normalisation, axis formatting, and file output separate makes this module
easier to inspect and maintain than one large plotting block per function.
"""

from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt
import mne
import numpy as np
from scipy.signal import welch


# ---------------------------------------------------------------------------
# Shared plotting configuration
# ---------------------------------------------------------------------------

# MNE accepts these channel-type names in ``pick`` and ``create_info``.  The
# set is used only to recognise an ndarray's requested channel type; Raw input
# keeps its channel metadata unchanged.
ALL_CHANNEL_LIST = frozenset({
    "grad", "mag", "eeg", "csd", "stim", "eog", "emg", "ecg",
    "ref_meg", "resp", "exci", "ias", "syst", "misc", "seeg", "dbs",
    "bio", "chpi", "dipole", "gof", "ecog", "hbo", "hbr", "temperature",
    "gsr", "eyetrack",
})


# The same small-font style is used by the public plots and by ``ckpt_report``.
_RC = {
    "font.size": 12,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
}

_DEFAULT_EVENT_COLORS = ("r", "g", "m", "c", "y", "k", "#7f7f7f")


# ---------------------------------------------------------------------------
# Generic figure and input helpers
# ---------------------------------------------------------------------------

def _unpack_plot(plot_out):
    """Return ``(figure, axes)`` for both MNE Spectrum plot return styles."""

    # MNE versions differ: some return a Figure, while others return
    # ``(Figure, axes)``.  Normalising here keeps the rest of psd_plot simple.
    if isinstance(plot_out, tuple):
        return plot_out
    return plot_out, plot_out.axes


def _axes_list(axes, fig):
    """Convert an MNE axes object or ndarray into a flat Python list."""

    if axes is None:
        return list(fig.axes)
    if isinstance(axes, np.ndarray):
        return list(axes.ravel())
    if isinstance(axes, (list, tuple)):
        return list(axes)
    return [axes]


def _array_channel_types(picks):
    """Choose channel types for an ndarray converted to an MNE RawArray."""

    # An ndarray has no channel names or metadata.  A single channel-type
    # string is therefore the only unambiguous metadata that can be retained.
    if isinstance(picks, str) and picks in ALL_CHANNEL_LIST:
        return picks
    return "eeg"


def _prepare_psd_input(eeg, fs, picks):
    """Copy/pick a Raw object, or convert an ndarray into a RawArray.

    Returns
    -------
    raw : mne.io.BaseRaw
        A private copy that can be modified by the plotting function.
    fs : float
        The sampling frequency used to calculate the requested FFT length.
    psd_picks : object
        Picks passed to ``compute_psd``.  Raw input is picked immediately, so
        ``'all'`` means every channel in the already-selected private copy.
    """

    # Step 1: copy and pick Raw input so the caller's object is never changed.
    if isinstance(eeg, mne.io.BaseRaw):
        raw = eeg.copy()
        if fs is None:
            fs = raw.info["sfreq"]
        if picks is not None:
            raw.pick(picks)
        # Picking first makes integer/name picks unambiguous.  Explicitly ask
        # compute_psd for all channels in this reduced Raw: picks=None makes
        # MNE silently fall back to EEG/MEG data channels and drops EOG/ECG.
        return raw, float(fs), "all"

    # Step 2: validate and wrap array-like input with minimal EEG metadata.
    data = np.asarray(eeg)
    if data.ndim != 2:
        raise ValueError(
            "Array input to psd_plot must have shape (channels, samples)."
        )
    if fs is None:
        raise ValueError("fs must be provided when eeg is not an MNE Raw object.")

    channel_names = ["ch_{}".format(index) for index in range(data.shape[0])]
    info = mne.create_info(
        channel_names,
        sfreq=float(fs),
        ch_types=_array_channel_types(picks),
    )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    return raw, float(fs), picks


def _make_finite_contiguous_raw(raw):
    """Remove non-finite samples and concatenate the remaining spans.

    MNE's ``reject_by_annotation='NaN'`` representation is useful for time
    plots, but it makes Welch split the recording into every good span.  Short
    spans then produce repeated ``nperseg is greater than input length``
    warnings.  PSD is insensitive to the original time gaps, so for this
    function we remove all columns containing a NaN/Inf and analyse the
    remaining samples as one contiguous signal.
    """

    # Step 1: explicitly request NaNs for bad annotations, rather than letting
    # MNE silently skip them in a version-dependent way.
    data = raw.get_data(reject_by_annotation="NaN")
    finite_columns = np.isfinite(data).all(axis=0)
    n_removed = int((~finite_columns).sum())

    if not finite_columns.any():
        raise ValueError("PSD calculation needs at least one finite sample.")

    # Step 2: always create a new RawArray.  This drops bad annotations and
    # therefore prevents MNE from splitting the cleaned signal again.
    cleaned = mne.io.RawArray(
        data[:, finite_columns],
        raw.info.copy(),
        first_samp=raw.first_samp,
        verbose="ERROR",
    )
    return cleaned, n_removed


def _compute_psd_with_retry(raw, fmin, fmax, n_fft, picks, dB, fs, debug=False):
    """Compute and plot a PSD, retaining the historical MNE fallbacks.

    The normal path has no NaNs because :func:`_make_finite_contiguous_raw`
    has already concatenated the valid samples.  The retry remains useful for
    older MNE/scipy combinations that report a NaN-related ValueError during
    plotting: it halves the FFT length until the calculation succeeds.
    """

    # Step 1: calculate the spectrum and ask MNE to render it.
    while True:
        try:
            spectrum = raw.compute_psd(
                fmin=fmin,
                fmax=fmax,
                n_fft=n_fft,
                picks=picks,
                reject_by_annotation=False,
            )
            try:
                plot_out = spectrum.plot(
                    dB=dB,
                    amplitude=True,
                    show=False,
                    picks=picks,
                )
            except RuntimeError as error:
                # A non-standard channel type may not be plottable with the
                # requested picks.  The spectrum itself is still valid, so
                # render it once without the pick restriction.
                if "No plottable channel types found" not in str(error):
                    raise
                plot_out = spectrum.plot(dB=dB, show=False)
            fig, axes = _unpack_plot(plot_out)
            return spectrum, fig, axes
        except ValueError as error:
            # This is a last-resort compatibility path.  Normal finite-data
            # preparation should make it unnecessary on current MNE versions.
            message = str(error)
            if "NaN" not in message or n_fft <= 16:
                raise
            n_fft = max(16, n_fft // 2)
            if debug:
                print(
                    "PSD retry with {:.6g} Hz/bin.".format(float(fs) / n_fft)
                )


def _format_psd_axes(fig, axes, dB):
    """Apply common PSD axis formatting after MNE has created the figure."""

    # Step 1: remove MNE's per-channel-type titles; the caller's ``name`` is
    # more useful as the single figure title.
    for axis in _axes_list(axes, fig):
        axis.set_title("")

    # Step 2: in dB mode, label only axes that MNE identified as spectral
    # axes.  Montage inset axes normally have an empty ylabel and stay empty.
    if dB:
        for axis in fig.axes:
            if axis.get_ylabel():
                axis.set_ylabel("dB")


def _set_figure_title(fig, axes, name):
    """Place a title over the main plotting area, if a name was supplied."""

    if name is None:
        return
    plot_axes = _axes_list(axes, fig)
    if not plot_axes:
        fig.suptitle(str(name))
        return

    # The montage inset can make the full-figure centre look off.  Centre over
    # the union of MNE's plotted axes instead.
    positions = [axis.get_position() for axis in plot_axes]
    left = min(position.x0 for position in positions)
    right = max(position.x0 + position.width for position in positions)
    fig.suptitle(str(name), x=left + 0.5 * (right - left))


def _save_or_show_psd(fig, save_pth):
    """Save a PSD as PDF or display it interactively."""

    if save_pth is None:
        plt.show()
        return

    # MNE's interactive RangeSlider registers draw callbacks that some static
    # backends cannot serialize.  Disconnect those callbacks before saving.
    for callback_id in list(
        fig.canvas.callbacks.callbacks.get("draw_event", {}).keys()
    ):
        fig.canvas.mpl_disconnect(callback_id)
    output_path = Path(save_pth).with_suffix(".pdf")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PSD plotting
# ---------------------------------------------------------------------------

def psd_plot(
    eeg,
    name=None,
    fs=None,
    picks="eeg",
    fmin=0,
    fmax=60,
    resolution=0.05,
    figsize=(20, 3),
    save_pth=None,
    debug=False,
    dB=False,
    rc=_RC,
):
    """Plot the power spectral density of one EEG dataset.

    Parameters
    ----------
    eeg : mne.io.Raw or array-like, shape (channels, samples)
        Data to plot. Raw input is copied before channel picking.
    name : str, optional
        Figure title.
    fs : float, optional
        Sampling frequency for array input. For Raw input the value defaults
        to ``eeg.info['sfreq']``.
    picks : str or list, optional
        Channels to include. Defaults to ``'eeg'``.
    fmin, fmax : float
        Frequency range in Hz.
    resolution : float
        Requested frequency-bin width in Hz. If the cleaned recording is too
        short, the FFT length is reduced to the available number of samples.
    figsize : tuple
        Figure size passed to Matplotlib.
    save_pth : path-like, optional
        Base path. PSD figures are written as PDF; ``None`` displays them.
    debug : bool
        Print the number of removed samples and any compatibility retry.
    dB : bool
        Plot in decibels. The spectral y-axis is labelled ``dB``.
    rc : dict, optional
        Temporary Matplotlib rc parameters.
    """

    if resolution <= 0:
        raise ValueError("resolution must be greater than zero.")

    # Step 1: make a private Raw object and resolve its sampling frequency.
    raw, fs, psd_picks = _prepare_psd_input(eeg, fs, picks)

    # Step 2: remove NaN/Inf columns and concatenate all valid spans.  This is
    # the key bad-segment fix: Welch sees one contiguous signal, not many
    # short spans that trigger repeated nperseg warnings.
    raw, n_removed = _make_finite_contiguous_raw(raw)
    if debug and n_removed:
        print(
            "PSD: removed and concatenated {} non-finite samples.".format(
                n_removed
            )
        )

    if raw.n_times < 2:
        raise ValueError("PSD calculation needs at least two finite samples.")

    # Step 3: convert the requested frequency resolution to an FFT length.
    requested_n_fft = max(2, int(np.round(float(fs) / resolution)))
    n_fft = min(requested_n_fft, raw.n_times)
    if debug and n_fft != requested_n_fft:
        print(
            "PSD: recording is shorter than the requested FFT; using n_fft={}.".format(
                n_fft
            )
        )

    # Step 4: calculate and render the spectrum while restoring MNE's global
    # log level even if MNE raises an exception.
    old_level = mne.set_log_level(
        "INFO" if debug else "ERROR", return_old_level=True
    )
    try:
        with matplotlib.rc_context(rc):
            spectrum, fig, axes = _compute_psd_with_retry(
                raw,
                fmin,
                fmax,
                n_fft,
                psd_picks,
                dB,
                fs,
                debug=debug,
            )
            _format_psd_axes(fig, axes, dB)
            fig.set_size_inches(figsize)
            _set_figure_title(fig, axes, name)
            _save_or_show_psd(fig, save_pth)
    finally:
        mne.set_log_level(old_level)

    return spectrum


# ---------------------------------------------------------------------------
# Time-domain plotting helpers
# ---------------------------------------------------------------------------

def _amplitude_yticks(seg, ylim=None, candidate_tick_counts=(7, 9, 11)):
    """Return readable, symmetric amplitude ticks for a signal segment.

    The signal is displayed after factoring out a power of ten.  For example,
    values around ``3e-5`` are labelled ``-3.00 ... 3.00`` with a y-axis label
    containing ``x 10^-5 V``.  The helper is kept independent of Matplotlib so
    it can be unit-tested directly.
    """

    # Step 1: determine finite data limits (or honour an explicit ylim).
    segment = np.asarray(seg, dtype=float)
    finite = segment[np.isfinite(segment)]
    if ylim is None:
        if finite.size:
            raw_min = float(finite.min())
            raw_max = float(finite.max())
        else:
            raw_min, raw_max = 0.0, 0.0
    else:
        raw_min, raw_max = float(ylim[0]), float(ylim[1])
        if not np.isfinite([raw_min, raw_max]).all():
            raise ValueError("ylim must contain two finite values.")

    # Step 2: avoid a zero-height axis for flat or empty signals.
    if np.isclose(raw_min, raw_max):
        if np.isclose(raw_min, 0.0):
            raw_min, raw_max = -1.0, 1.0
        else:
            span = abs(raw_min) * 0.1
            raw_min, raw_max = raw_min - span, raw_max + span

    # Step 3: choose a power-of-ten factor that keeps displayed magnitudes in
    # a compact range, then test the requested tick counts.
    maximum = max(abs(raw_min), abs(raw_max))
    if maximum <= 0 or not np.isfinite(maximum):
        exponent = 0
    else:
        exponent = int(np.floor(np.log10(maximum)))
        while maximum / (10 ** exponent) < 1:
            exponent -= 1
        while maximum / (10 ** exponent) >= 10:
            exponent += 1

    factor = 10 ** exponent
    raw_min_scaled = raw_min / factor
    raw_max_scaled = raw_max / factor
    centre_scaled = round(0.5 * (raw_min_scaled + raw_max_scaled), 2)

    def round_step_up(value):
        return np.ceil(value * 100.0) / 100.0

    candidates = []
    for n_ticks in candidate_tick_counts:
        if n_ticks < 3 or n_ticks % 2 == 0:
            raise ValueError("candidate_tick_counts must contain odd values >= 3.")
        half = (n_ticks - 1) / 2.0
        needed = max(
            (centre_scaled - raw_min_scaled) / half,
            (raw_max_scaled - centre_scaled) / half,
            0.0,
        )
        step = round_step_up(max(needed, 0.01))
        tick_min = centre_scaled - half * step
        tick_max = centre_scaled + half * step
        covers = tick_min <= raw_min_scaled + 1e-12
        covers = covers and tick_max >= raw_max_scaled - 1e-12
        covers = covers and tick_min > -10.0 and tick_max < 10.0
        if covers:
            candidates.append(
                {
                    "n": n_ticks,
                    "step": step,
                    "tmin": tick_min,
                    "tmax": tick_max,
                    "ylim": (tick_min * factor, tick_max * factor),
                    "span": (tick_max - tick_min) * factor,
                }
            )

    # Step 4: use the tightest valid candidate, or a conservative seven-tick
    # fallback when the signal spans an unusually large scaled range.
    chosen = min(candidates, key=lambda candidate: candidate["span"]) if candidates else None
    if chosen is None:
        n_ticks = 7
        step = max(
            0.01,
            round_step_up((raw_max_scaled - raw_min_scaled) / (n_ticks - 1)),
        )
        tick_min = np.floor(round(raw_min_scaled, 2) * 100.0) / 100.0
        tick_max = tick_min + (n_ticks - 1) * step
        chosen = {
            "n": n_ticks,
            "step": step,
            "tmin": tick_min,
            "tmax": tick_max,
            "ylim": (tick_min * factor, tick_max * factor),
        }

    # Step 5: return raw-unit tick positions and scaled labels.
    indexes = np.arange(chosen["n"])
    display_ticks = np.round(
        chosen["tmin"] + indexes * chosen["step"], 2
    )
    tick_values = display_ticks * factor
    tick_labels = ["{:.2f}".format(value) for value in display_ticks]
    return tick_values, tick_labels, chosen["ylim"], exponent


def _plot_data(eeg, fs):
    """Return ``(data, fs, channel_names, first_samp)`` for plot functions."""

    if isinstance(eeg, mne.io.BaseRaw):
        sample_rate = eeg.info["sfreq"] if fs is None else fs
        # ``NaN`` keeps bad annotations visible as gaps in time plots.
        data = eeg.get_data(reject_by_annotation="NaN")
        # Events supplied by the SEMP tutorial are relative to this displayed
        # data array, so use a zero-based sample origin here.
        first_samp = 0
        return data, float(sample_rate), list(eeg.ch_names), first_samp

    data = np.asarray(eeg)
    if data.ndim != 2:
        raise ValueError("Array input must have shape (channels, samples).")
    sample_rate = 5000.0 if fs is None else float(fs)
    channel_names = ["ch_{}".format(index) for index in range(data.shape[0])]
    return data, sample_rate, channel_names, 0


def _channel_index(channel, channel_names, n_channels):
    """Resolve a channel name or integer index with a useful error message."""

    if isinstance(channel, str):
        if channel not in channel_names:
            raise ValueError(
                "Channel {!r} is not present in the input.".format(channel)
            )
        return channel_names.index(channel)

    index = int(channel)
    if index < 0 or index >= n_channels:
        raise IndexError(
            "Channel index {} is outside the range [0, {}).".format(
                index, n_channels
            )
        )
    return index


def _plot_window(start, length, n_samples):
    """Validate a sample window and clamp its end to available data."""

    start = int(start)
    if start < 0:
        raise ValueError("start must be non-negative.")
    if start >= n_samples:
        raise ValueError(
            "start={} is outside the data with {} samples.".format(
                start, n_samples
            )
        )
    if length is None:
        length = n_samples - start
    length = int(length)
    if length <= 0:
        raise ValueError("length must be a positive number of samples.")
    stop = min(start + length, n_samples)
    return start, stop


def _normalise_event_series(events, event_id, event_name):
    """Convert single or multiple event inputs into parallel lists."""

    if events is None:
        return [], [], []

    # A 2-D ndarray (or 2-D list) is one MNE-style event array.  A list of
    # arrays is instead interpreted as multiple event series.  Looking at the
    # first element avoids ``np.asarray`` collapsing a list of equal-shaped
    # arrays into an ambiguous 3-D object array.
    if isinstance(events, np.ndarray):
        event_list = list(events) if events.ndim >= 3 else [events]
    elif isinstance(events, (list, tuple)) and events:
        first = events[0]
        if isinstance(first, np.ndarray):
            event_list = list(events)
        elif isinstance(first, (list, tuple)) and first:
            # ``[[sample, previous, code], ...]`` is one series; nested rows
            # such as ``[array(...), array(...)]`` are multiple series.
            first_value = first[0]
            if isinstance(first_value, (list, tuple, np.ndarray)):
                event_list = list(events)
            else:
                event_list = [events]
        else:
            event_list = [events]
    else:
        event_list = [events]

    if isinstance(event_id, (list, tuple, np.ndarray)) and len(event_list) > 1:
        id_list = list(event_id)
    else:
        id_list = [event_id] * len(event_list)

    if isinstance(event_name, (list, tuple)) and len(event_name) == len(event_list):
        name_list = list(event_name)
    else:
        name_list = [event_name] * len(event_list)
    return event_list, id_list, name_list


def _event_samples(event_array, event_id, first_samp, start, stop):
    """Return event sample positions in a requested displayed window."""

    if event_array is None:
        return []
    values = np.asarray(event_array)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        # A single MNE event row has [sample, previous, code].  A one-column
        # array is treated as a sequence of sample positions.
        values = values.reshape(1, -1) if values.size >= 3 else values.reshape(-1, 1)

    samples = []
    for event in values:
        try:
            sample = int(event[0]) - first_samp
            code = event[2] if len(event) > 2 else None
        except (TypeError, ValueError, IndexError):
            continue
        if start <= sample < stop and (event_id is None or code == event_id):
            samples.append(sample)
    return samples


def _draw_event_raster(
    events,
    event_id,
    event_name,
    start,
    length,
    fs,
    first_samp,
    event_onset,
    colors,
    ax=None,
):
    """Draw event markers in rows below a time-domain trace."""

    # Step 1: normalise single/list event inputs and select the current axes.
    event_list, id_list, name_list = _normalise_event_series(
        events, event_id, event_name
    )
    if ax is None:
        ax = plt.gca()
    colors = tuple(colors) if colors is not None else _DEFAULT_EVENT_COLORS
    stop = start + length

    # Step 2: reserve a small band below the signal for event rows.
    y0, y1 = ax.get_ylim()
    y_span = max(y1 - y0, np.finfo(float).eps)
    row_gap = 0.03 * y_span
    row_step = 0.02 * y_span
    n_rows = max(len(event_list), 1)
    if n_rows * row_step + row_gap > 0.20 * y_span:
        row_step = max(0.001 * y_span, (0.20 * y_span - row_gap) / n_rows)

    plotted = False
    row_positions = []
    for row, (event_array, selected_id, label) in enumerate(
        zip(event_list, id_list, name_list)
    ):
        samples = _event_samples(
            event_array, selected_id, first_samp, start, stop
        )
        if not samples:
            continue
        y_row = y0 - row_gap - row * row_step
        row_positions.append(y_row)
        label = str(label) if label is not None else "event_{}".format(row)
        ax.scatter(
            [sample / fs + event_onset for sample in samples],
            [y_row] * len(samples),
            marker="^",
            s=60,
            facecolor=colors[row % len(colors)],
            edgecolors="k",
            linewidths=0.4,
            zorder=6,
            label=label,
            clip_on=False,
        )
        plotted = True

    # Step 3: extend the lower limit so markers remain visible and add one
    # legend entry per event series.
    if plotted:
        fig_height = ax.figure.get_size_inches()[1]
        marker_height = np.sqrt(60.0) / 72.0
        pad = max(0.01 * y_span, marker_height * y_span / fig_height * 0.6)
        ax.set_ylim(min(row_positions) - pad, y1)
        ax.legend(loc="upper right")


def _save_or_show(fig, save_pth, **save_kwargs):
    """Save a normal Matplotlib figure or display it interactively."""

    if save_pth is None:
        plt.show()
    else:
        fig.savefig(save_pth, **save_kwargs)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Time-domain plots
# ---------------------------------------------------------------------------

def temp_plot(
    eeg,
    channel,
    start=0,
    length=None,
    fs=None,
    events=None,
    event_id=None,
    event_onset=0,
    name=None,
    save_pth=None,
    figsize=(20, 3),
    ylim=None,
    event_name=None,
    rc=_RC,
    candidate_tick_counts=(7, 9, 11),
    colors=_DEFAULT_EVENT_COLORS,
):
    """Plot one channel in the time domain, optionally with event markers."""

    # Step 1: extract data and resolve the requested channel/window.
    data, fs, channel_names, first_samp = _plot_data(eeg, fs)
    channel_index = _channel_index(channel, channel_names, data.shape[0])
    start, stop = _plot_window(start, length, data.shape[1])
    title = str(channel) if name is None else str(name)
    event_name = "event" if event_name is None else event_name

    with matplotlib.rc_context(rc):
        # Step 2: draw the selected trace and configure readable amplitude
        # ticks, labels, limits, and title.
        fig, ax = plt.subplots(figsize=figsize)
        segment = data[channel_index, start:stop]
        ax.plot(np.arange(start, stop) / fs, segment)
        tick_values, tick_labels, selected_ylim, exponent = _amplitude_yticks(
            segment,
            ylim=ylim,
            candidate_tick_counts=candidate_tick_counts,
        )
        ax.set_yticks(tick_values, tick_labels)
        ax.set_ylim(selected_ylim)
        ax.set_ylabel("Amplitude ($\\times 10^{{{}}}$ V)".format(exponent))
        ax.set_xlabel("time (s)")
        ax.set_xlim(start / fs, stop / fs)
        ax.set_title(title)

        # Step 3: add event rows below the trace and save/display the figure.
        _draw_event_raster(
            events,
            event_id,
            event_name,
            start,
            stop - start,
            fs,
            first_samp,
            event_onset,
            colors,
            ax=ax,
        )
        _save_or_show(fig, save_pth, bbox_inches="tight", pad_inches=0.01)


def _aligned_plot_data(eeg, fs):
    """Return plot data plus its absolute sample origin and channel names."""

    data, sample_rate, channel_names, first_samp = _plot_data(eeg, fs)
    if isinstance(eeg, mne.io.BaseRaw):
        # Unlike temp_plot, diff plots align two recordings in their absolute
        # sample coordinate system before trimming to their overlap.
        first_samp = int(eeg.first_samp)
    return data, sample_rate, channel_names, int(first_samp)


def temp_plot_diff(
    eeg,
    eeg2,
    channel,
    start=0,
    length=None,
    fs=None,
    events=None,
    event_id=None,
    plot_eeg=False,
    event_onset=0,
    name="BCG removal",
    save_pth=None,
    figsize=(20, 3),
):
    """Plot two aligned traces or their sample-wise difference."""

    # Step 1: load both inputs and resolve the channel in each recording.
    data1, fs1, names1, first1 = _aligned_plot_data(eeg, fs)
    data2, fs2, names2, first2 = _aligned_plot_data(eeg2, fs)
    if not np.isclose(fs1, fs2):
        raise ValueError("eeg and eeg2 must have the same sampling frequency.")
    channel1 = _channel_index(channel, names1, data1.shape[0])
    channel2 = _channel_index(channel, names2, data2.shape[0])

    # Step 2: trim both arrays to their overlapping absolute sample range.
    overlap_start = max(first1, first2)
    offset1 = overlap_start - first1
    offset2 = overlap_start - first2
    n_samples = min(data1.shape[1] - offset1, data2.shape[1] - offset2)
    if n_samples <= 0:
        raise ValueError("eeg and eeg2 have no overlapping samples.")
    data1 = data1[:, offset1:offset1 + n_samples]
    data2 = data2[:, offset2:offset2 + n_samples]
    start, stop = _plot_window(start, length, n_samples)

    # Step 3: draw either both traces or their aligned difference.
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(start, stop) / fs1
    if plot_eeg:
        ax.plot(x, data1[channel1, start:stop], label="Before")
        ax.plot(x, data2[channel2, start:stop], label="After", color="orange")
    else:
        difference = data2[channel2] - data1[channel1]
        ax.plot(x, difference[start:stop], label="Difference")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Amplitude (V)")
    ax.set_title("Before and After {}".format(name))

    # Step 4: mark selected events using the overlap's sample origin.
    if events is not None and event_id is not None:
        samples = _event_samples(
            events, event_id, overlap_start, start, stop
        )
        for index, sample in enumerate(samples):
            ax.axvline(
                sample / fs1 + event_onset,
                color="r",
                label="event" if index == 0 else None,
            )
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    _save_or_show(fig, save_pth)


# ---------------------------------------------------------------------------
# PCA component plots
# ---------------------------------------------------------------------------

def _plot_pc_panel(component_data, channel_name, title_prefix, fs, n_fft, psd_lim, output_path, figsize):
    """Write one component's time/PSD panel for one channel and window."""

    n_components = component_data.shape[-1]
    x = np.arange(component_data.shape[-2]) / fs
    fig, axes = plt.subplots(n_components, 2, figsize=figsize, squeeze=False)

    # Each row contains one component's time course and Welch PSD.
    for component in range(n_components):
        signal = np.asarray(component_data[..., component], dtype=float)
        axes[component, 0].plot(
            x,
            signal,
            label="PC{}".format(component),
        )
        axes[component, 0].set_title(
            "PC{} for {}".format(component, title_prefix)
        )
        axes[component, 0].set_xlim(x[0], x[-1])
        axes[component, 0].legend()

        # PCA output should be finite; if an upstream bad segment remains,
        # remove those samples here rather than passing NaNs to scipy.welch.
        finite = np.isfinite(signal)
        signal_for_psd = signal[finite]
        if signal_for_psd.size >= 2:
            freqs, power = welch(
                signal_for_psd,
                fs,
                nperseg=min(n_fft, signal_for_psd.size),
            )
            axes[component, 1].plot(freqs, power, label="PSD")
        axes[component, 1].set_title(
            "PC{} PSD for {}".format(component, title_prefix)
        )
        axes[component, 1].set_xlim(psd_lim)
        axes[component, 1].legend()

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def pcs_plot(
    pcs,
    target_fdr,
    ch_list,
    ch_names,
    info,
    win_list=None,
    figsize=(20, 9),
    resolution=0.05,
    psd_lim=(0, 50),
):
    """Save time/PSD panels for PCA components.

    ``pcs`` may have shape ``(channels, samples, components)`` or
    ``(windows, channels, samples, components)``.  ``target_fdr`` is retained
    as the historical parameter name for API compatibility and denotes the
    output directory.
    """

    if resolution <= 0:
        raise ValueError("resolution must be greater than zero.")
    pcs = np.asarray(pcs)
    if pcs.ndim not in (3, 4):
        raise ValueError(
            "pcs should have shape (channels, samples, components) or "
            "(windows, channels, samples, components)."
        )

    # Step 1: validate the complete, ordered name mapping against the PCA
    # channel axis. Bad channels must not be removed before this check: doing
    # so shifts every later name-to-row mapping and can silently mislabel PCs.
    fs = float(info["sfreq"])
    ch_names = list(ch_names)
    if len(set(ch_names)) != len(ch_names):
        raise ValueError("ch_names contains duplicates and is not a safe mapping.")
    bad_channels = set(info.get("bads", []))
    n_pcs_channels = pcs.shape[-3]
    if len(ch_names) != n_pcs_channels:
        raise ValueError(
            "Number of channels in pcs ({}) does not match its ordered "
            "channel-name mapping ({}).".format(n_pcs_channels, len(ch_names))
        )

    requested_channels = [
        name for name in ch_list
        if name not in bad_channels and name in ch_names
    ]
    if not requested_channels:
        raise ValueError("ch_list contains no usable channel names.")

    # Step 2: choose the window(s) to plot and create the output directory.
    output_dir = Path(target_fdr)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_fft = max(2, int(np.round(fs / resolution)))
    if pcs.ndim == 3:
        windows = [None]
    elif win_list is None:
        windows = list(np.random.choice(np.arange(pcs.shape[0]), 1))
    else:
        windows = list(win_list)

    for window in windows:
        # Step 3: select one window (or the only 3-D PCA array) and write one
        # panel per requested channel.
        for channel_name in requested_channels:
            channel_index = ch_names.index(channel_name)
            if pcs.ndim == 3:
                component_data = pcs[channel_index]
                output_name = "pc_{}.png".format(channel_name)
                title_prefix = "channel {}".format(channel_name)
            else:
                if int(window) < 0 or int(window) >= pcs.shape[0]:
                    raise IndexError(
                        "PCA window {} is outside the available range.".format(
                            window
                        )
                    )
                component_data = pcs[int(window), channel_index]
                output_name = "pc_{}_win_{}.png".format(channel_name, window)
                title_prefix = "window {} of channel {}".format(
                    window, channel_name
                )
            _plot_pc_panel(
                component_data,
                channel_name,
                title_prefix,
                fs,
                n_fft,
                psd_lim,
                output_dir / output_name,
                figsize,
            )
