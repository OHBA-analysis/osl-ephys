"""User-facing visualization for semp.

These are the plotting helpers you call directly (in a notebook, or from a
project script) to look at the data: single-recording PSDs, time-domain traces,
before/after diffs, and the AAS/OBS component panels. ``ckpt_report`` uses the
same functions to build its per-checkpoint figures.

Non-plotting support code lives under ``semp.utils`` (io / signal / config);
metric primitives live in ``semp.metric``.
"""
import os, copy
import numpy as np
import mne
import matplotlib
from matplotlib import pyplot as plt
from scipy.signal import welch

ALL_CHANNEL_LIST = {'grad', 'mag', 'eeg', 'csd', 'stim', 'eog', 'emg', 'ecg', 'ref_meg', 'resp', 'exci', 'ias', 'syst', 'misc', 'seeg', 'dbs', 'bio', 'chpi', 'dipole', 'gof', 'ecog', 'hbo', 'hbr', 'temperature', 'gsr', 'eyetrack'}  # https://mne.tools/1.6/generated/mne.channel_type.html

#: Shared matplotlib rc for psd_plot / temp_plot (small font, sans-serif). One
#: module constant instead of an identical dict literal defaulted on each func.
_RC = {
    'font.size': 12,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
}


def _unpack_plot(plot_out):
    """Normalise MNE's ``Spectrum.plot()`` return to ``(fig, axes)``.

    Depending on the MNE version it returns either a bare ``Figure`` or a
    ``(fig, axes)`` tuple; this collapses both to ``(fig, axes)``.
    """
    if isinstance(plot_out, tuple):
        return plot_out
    return plot_out, plot_out.axes


def _compute_psd_with_retry(raw, fmin, fmax, n_fft, picks, dB, fs):
    """Compute + plot a PSD, with the two fallbacks psd_plot needs.

    * On a ``NaN`` ``ValueError`` (resolution too fine for the data) halve
      ``n_fft`` and retry, floored so it can't reach 0.
    * On MNE's "No plottable channel types found" ``RuntimeError`` re-plot the
      (already computed) PSD without ``picks``.

    Returns ``(psd, fig, axes)``. This is the fragile control flow lifted out of
    ``psd_plot`` verbatim so the plotting body stays linear.
    """
    while True:
        try:
            psd = raw.compute_psd(fmin=fmin, fmax=fmax, n_fft=n_fft, picks=picks)
            plot_out = psd.plot(dB=dB, amplitude=True, show=False, picks=picks)
            fig, axes = _unpack_plot(plot_out)
            return psd, fig, axes
        except ValueError as e:
            if 'NaN' in str(e) and n_fft // 2 >= 16:
                n_fft = n_fft // 2   # coarser resolution; floored so it can't reach 0
                print(f'WARNING: PSD calculation failed, trying again with a smaller resolution {int(fs / n_fft)} Hz/bin')
            else:
                raise e
        except RuntimeError as e:
            if 'No plottable channel types found' in str(e):
                plot_out = psd.plot(dB=dB, show=False)   # single render, no picks
                fig, axes = _unpack_plot(plot_out)
                return psd, fig, axes
            else:
                raise e


def psd_plot(eeg, name=None, fs=None, picks='eeg', fmin=0, fmax=60, resolution=0.05, figsize=(20,3), save_pth=None, debug=False, dB=False, rc=_RC):
    """
    Plot the power spectral density (PSD) of a single EEG dataset.

    Parameters
    ----------
    eeg : mne.io.Raw or ndarray-like
        EEG data. If ndarray-like, shape should be (channels, time).
    name : str, optional
        Title or filename suffix. Default: None.
    fs : float, optional
        Sampling frequency. If None, taken from Raw, or raised as an error.
    picks : str or list, optional
        Channels to include. Default: 'eeg'.
    fmin, fmax : float
        Frequency range. Default: 0–60 Hz.
    resolution : float
        Resolution for psd map. Default: 0.05 Hz/bin.
    figsize : tuple
        Matplotlib figure size. Default: (20, 3).
    save_pth : str, optional
        Base path to save figure. If None, just show.
    debug : bool
        If True, print debug info.
    dB : bool
        Whether to plot in dB scale.
    """
    verbose = 'INFO' if debug else 'ERROR'
    eeg = copy.deepcopy(eeg)  # avoid modifying original data (e.g., picking channels in-place)
    # matplotlib.rcParams.update(rc)
    if 'mne.io' in str(type(eeg)):  # already Raw
        if fs is None:
            fs = eeg.info['sfreq']
        # raw = pick_indices(eeg, picks, return_indices=False)
        raw = eeg.pick(picks)
    else:  # numpy or torch
        eeg = np.array(eeg)
        if fs is None:
            raise ValueError("fs must be provided if eeg is not an mne.io.Raw object.")
        ch_types = picks if picks in ALL_CHANNEL_LIST else 'eeg'
        raw = mne.io.RawArray(eeg, mne.create_info(eeg.shape[0], sfreq=fs, ch_types=ch_types))

    n_fft = int(np.round(fs / resolution))

    old_level = mne.set_log_level(verbose, return_old_level=True)

    with matplotlib.rc_context(rc):
        psd, fig, axes = _compute_psd_with_retry(raw, fmin, fmax, n_fft, picks, dB, fs)
        # --- Suppress MNE's per-panel channel-type titles (e.g., "EEG") ---
        try:
            for ax in (axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]):
                ax.set_title('')
        except Exception:
            for ax in fig.axes:
                ax.set_title('')
        fig.set_size_inches(figsize)
        if name is not None:
            # after plotting and getting `fig` and `axes`
            axes_list = axes if isinstance(axes, (list, tuple, np.ndarray)) else [axes]

            # get per-axis positions in figure coordinates
            pos_list = [ax.get_position() for ax in axes_list]

            # combined left and right edge of the plotting area
            left = min(p.x0 for p in pos_list)
            right = max(p.x0 + p.width for p in pos_list)

            center_fig = left + 0.5 * (right - left)
            fig.suptitle(str(name), x=center_fig)   # tweak y to taste

        try:
            if save_pth is not None:
                # MNE's psd.plot() attaches a RangeSlider that registers a draw_event
                # callback using copy_from_bbox (blitting), which is unsupported by
                # non-interactive backends like PDF. Disconnect before saving.
                for cid in list(fig.canvas.callbacks.callbacks.get('draw_event', {}).keys()):
                    fig.canvas.mpl_disconnect(cid)
                fig.savefig(save_pth.with_suffix(".pdf"), bbox_inches='tight', pad_inches=0)
                plt.close(fig)
            else:
                plt.show()
        except ValueError as e:
            print(f"WARNING: {e}. Plotting without saving to file.")
            print(f"Current psd is : {psd}")

    mne.set_log_level(old_level)
    return psd

def _amplitude_yticks(seg, ylim=None, candidate_tick_counts=(7, 9, 11)):
    """Symmetric, 2-decimal y-ticks with a factored 10**exp scale (temp_plot's
    amplitude axis, extracted verbatim).

    An exponent is factored out so the signal magnitude sits in [1, 10); ticks
    are symmetric around the 2-decimal midpoint with a 0.01-multiple step, using
    the tick count in ``candidate_tick_counts`` that covers the data with the
    smallest span (fallback: 7 ticks spanning the data). Returns
    ``(ytick_vals, ytick_labels, ylim, exp)`` -- ``ytick_vals``/``ylim`` in
    raw-data units, ``ytick_labels`` the 2-decimal scaled strings.
    """
    seg = np.asarray(seg, dtype=float)
    if ylim is None:
        raw_min = np.nanmin(seg) if seg.size > 0 else 0.0
        raw_max = np.nanmax(seg) if seg.size > 0 else 0.0
    else:
        raw_min, raw_max = float(ylim[0]), float(ylim[1])

    # degenerate / zero-range signals
    if np.isclose(raw_min, raw_max):
        if np.isclose(raw_min, 0.0):
            raw_min, raw_max = -1.0, 1.0
        else:
            span = abs(raw_min) * 0.1 if abs(raw_min) > 0 else 1.0
            raw_min, raw_max = raw_min - span, raw_max + span

    med = 0.5 * (raw_min + raw_max)

    # exponent so the signal magnitude sits in [1, 10)
    try:
        vmax_abs = float(np.nanmax([abs(raw_min), abs(raw_max)]))
    except Exception:
        vmax_abs = 0.0
    if np.isnan(vmax_abs) or vmax_abs <= 0.0:
        exp = 0
    else:
        exp = int(np.floor(np.log10(vmax_abs)))
        while True:
            vmax_s = vmax_abs / (10 ** exp)
            if vmax_s < 1:
                exp -= 1
                continue
            if vmax_s >= 10:
                exp += 1
                continue
            break

    factor = 10 ** exp
    raw_min_s, raw_max_s, med_s = raw_min / factor, raw_max / factor, med / factor
    center_s = round(med_s, 2)

    def step_round_up(x):
        return np.ceil(x * 100.0) / 100.0

    # candidate tick counts: pick the one that covers the data with least span
    candidates = []
    for n_ticks in candidate_tick_counts:
        half = (n_ticks - 1) / 2.0
        step_needed = max((center_s - raw_min_s) / half, (raw_max_s - center_s) / half, 0.0)
        if step_needed <= 0:
            step_needed = 0.01
        step = step_round_up(step_needed)
        tmin = center_s - half * step
        tmax = center_s + half * step
        covers = (tmin <= raw_min_s + 1e-12) and (tmax >= raw_max_s - 1e-12)
        if np.any(np.isclose(np.array([tmin, tmax]), -10.0, atol=1e-9)) or np.any(np.isclose(np.array([tmin, tmax]), 10.0, atol=1e-9)):
            covers = False
        if tmin <= -10.0 + 1e-12 or tmax >= 10.0 - 1e-12:
            covers = False
        if covers:
            ylim_low_raw, ylim_high_raw = tmin * factor, tmax * factor
            candidates.append({'n': n_ticks, 'step': step, 'tmin': tmin, 'tmax': tmax,
                               'ylim': (ylim_low_raw, ylim_high_raw),
                               'span': ylim_high_raw - ylim_low_raw})

    chosen = sorted(candidates, key=lambda x: x['span'])[0] if candidates else None

    if chosen is None:   # fallback: 7 ticks spanning the data
        n_ticks = 7
        if raw_max_s - raw_min_s <= 0:
            step = 0.01
        else:
            step = max(0.01, step_round_up((raw_max_s - raw_min_s) / (n_ticks - 1)))
        tmin = int(np.floor(round(raw_min_s, 2) * 100.0)) / 100.0
        tmax = tmin + (n_ticks - 1) * step
        chosen = {'n': n_ticks, 'step': step, 'tmin': tmin, 'tmax': tmax,
                  'ylim': (tmin * factor, tmax * factor),
                  'span': (tmax - tmin) * factor}

    idxs = np.arange(0, chosen['n'])
    disp_ticks = np.round(chosen['tmin'] + idxs * chosen['step'], 2)
    ytick_vals = disp_ticks * factor
    ytick_labels = [f"{v:.2f}" for v in disp_ticks]
    return ytick_vals, ytick_labels, chosen['ylim'], exp


def _draw_event_raster(events, event_id, event_name, start, length, fs,
                       first_samp, event_onset, colors):
    """Draw temp_plot's event markers as colored rows below the trace.

    Operates on the *current* pyplot axes (temp_plot has just plotted the trace
    and set its ylim): places one raster row per event series in the margin
    below the signal, then extends the bottom ylim to fit the lowest row plus a
    marker-size pad. Behaviour is identical to the inline block it replaces --
    this is a pure extraction so temp_plot reads as "plot trace / axes / raster".
    """
    # normalize events/event_id into parallel lists (support both single and list forms)
    if isinstance(events, (list, tuple)) and isinstance(event_id, (list, tuple)):
        ev_list = list(events)
        id_list = list(event_id)
        if isinstance(event_name, (list, tuple)) and len(event_name) == len(ev_list):
            names = list(event_name)
        else:
            names = [event_name] * len(ev_list)
    else:
        ev_list = [events]
        id_list = [event_id]
        names = [event_name]

    # visual params
    marker = '^'
    marker_size = 60
    z = 6

    # compute positions for raster rows (placed below the plotted signal)
    y0, y1 = plt.ylim()
    y_span = y1 - y0
    row_gap = 0.03 * y_span             # gap between signal bottom and first raster row
    row_step = 0.02 * y_span            # vertical spacing between rows
    max_rows = len(ev_list)
    # don't allow raster to extend >20% of plot height; compress if necessary
    if max_rows * row_step + row_gap > 0.20 * y_span:
        row_step = max(0.20 * y_span - row_gap, 0.001 * y_span) / max(1, max_rows)

    any_plotted = False
    for i, (ev_arr, eid, nm) in enumerate(zip(ev_list, id_list, names)):
        if ev_arr is None:
            continue
        try:
            evs = ev_arr.copy()
        except Exception:
            evs = np.array(ev_arr)
        xs = []
        for ev in evs:
            # support both Nx>=1 arrays (ev[0] sample, ev[2] code) and scalar event formats
            try:
                samp = int(ev[0]) - first_samp
                code = ev[2] if len(ev) > 2 else None
            except Exception:
                try:
                    samp = int(ev) - first_samp
                    code = None
                except Exception:
                    continue
            if samp > start and samp < start + length:
                if (eid is None) or (code == eid):
                    xs.append(samp / fs + event_onset)
        if len(xs) == 0:
            continue
        # y position for this row (below plot)
        y_row = y0 - row_gap - i * row_step
        # label: use provided name (only once per series)
        label = names[i] if isinstance(names[i], str) else f'event_{i}'
        # draw markers but avoid clipping (also add a thin black edge for contrast)
        plt.scatter(xs, [y_row] * len(xs),
                    marker=marker, s=marker_size,
                    facecolor=colors[i % len(colors)],
                    edgecolors='k', linewidths=0.4,
                    zorder=z, label=label,
                    clip_on=False)
        any_plotted = True

        # ensure markers are not visually cropped: pad bottom ylim based on marker size (points -> data units)
        fig = plt.gcf()
        fig_h_in = fig.get_size_inches()[1]           # figure height in inches
        marker_diam_pts = np.sqrt(marker_size)        # marker "diameter" in points (approx)
        marker_h_in = marker_diam_pts / 72.0          # convert points -> inches (1 pt = 1/72 in)
        # data-units per inch on the y-axis:
        data_per_in = (y_span) / fig_h_in
        # pad in data units; 0.6 factor because triangular marker height < full diameter
        pad_data = marker_h_in * data_per_in * 0.6
        # make sure pad is at least a small fraction of the y-span
        pad_data = max(pad_data, 0.01 * y_span)

    if any_plotted:
        plt.legend(loc='upper right')
    # after looping all rows (once), extend ylim to include lowest row + padding
    if any_plotted:
        min_row = y0 - row_gap - (max_rows - 1) * row_step
        plt.ylim(min_row - pad_data, y1)


def temp_plot(eeg, channel, start=0, length=None, fs=None, events=None, event_id=None, event_onset=0, name=None, save_pth=None, figsize=(20,3), ylim=None, event_name=None, rc=_RC, candidate_tick_counts=[7,9,11], colors = ['r', 'g', 'm', 'c', 'y', 'k', '#7f7f7f']):
    name = name if name is not None else f'{channel}'
    event_name = event_name if event_name is not None else 'event'
    if 'mne.io' in str(type(eeg)):
        if fs is None:
            fs = eeg.info['sfreq']
        data = eeg.get_data(reject_by_annotation='NaN')
        # first_samp = eeg.first_samp
        first_samp = 0
        if isinstance(channel, str):
            channel = eeg.ch_names.index(channel)
    else:
        if fs is None:
            fs = 5000
        data = eeg.copy()
        first_samp = 0
    if length is None:
        length = data.shape[1]

    with matplotlib.rc_context(rc):
        plt.figure(figsize=figsize)
        start = int(start)
        length = int(length)

        plt.plot(np.arange(start, start + length)/fs, data[channel][start:start+length])

        # === y-axis: factored-exponent, symmetric 2-decimal ticks ===
        seg = data[channel][start:start+length]
        ytick_vals, ytick_labels, ylim_chosen, exp = _amplitude_yticks(
            seg, ylim=ylim, candidate_tick_counts=candidate_tick_counts)
        plt.yticks(ytick_vals, ytick_labels)
        plt.ylim(ylim_chosen)
        plt.ylabel(fr"Amplitude ($\times 10^{{{exp}}}$ V)")
        plt.xlabel('time (s)')
        plt.xlim(start / fs, (start + length) / fs)

        # --- event raster: markers in rows below the trace (see helper) ---
        _draw_event_raster(events, event_id, event_name, start, length, fs,
                           first_samp, event_onset, colors)

        if save_pth is not None:
            plt.savefig(save_pth, bbox_inches='tight', pad_inches=0.01)
            plt.close()
        else:
            plt.show()


def temp_plot_diff(eeg, eeg2, channel, start=0, length=None, fs=None, events=None, event_id=None, plot_eeg=False, event_onset=0, name='BCG removal', save_pth=None, figsize=(20,3)):
    first_samp = 0
    plt.figure(figsize=figsize)
    if not isinstance(channel, str):
        channel1 = channel2 = channel
    if 'mne.io' in str(type(eeg)):
        if fs is None:
            fs = eeg.info['sfreq']
        data = eeg.get_data(reject_by_annotation='NaN')
        first_samp = eeg.first_samp
        if isinstance(channel, str):
            channel1 = eeg.ch_names.index(channel)
    else:
        if fs is None:
            fs = 5000
        data = eeg.copy()
        first_samp = 0
    if 'mne.io' in str(type(eeg2)):
        data2 = eeg2.get_data(reject_by_annotation='NaN')
        # assert eeg2.first_samp == first_samp, "eeg and eeg2 should have the same first_samp"
        if eeg2.first_samp > first_samp:
            data = data[:, eeg2.first_samp-first_samp:]
        if eeg2.first_samp < first_samp:
            data2 = data2[:, first_samp-eeg2.first_samp:]

        if isinstance(channel, str):
            channel2 = eeg2.ch_names.index(channel)
    else:
        data2 = eeg2.copy()
    if length is None:
        length = min(data.shape[1], data2.shape[1])

    start = int(start)
    length = int(length)
    if plot_eeg:
        plt.plot(np.arange(start, start + length)/fs, data[channel1][start:start+length], label="Before")
        plt.plot(np.arange(start, start + length)/fs, data2[channel2][start:start+length], label="After", color="orange")
    else:
        data_total_length = min(data.shape[1], data2.shape[1])
        plt.plot(np.arange(start, start + length)/fs, (data2[channel2, :data_total_length]-data[channel1, :data_total_length])[start:start+length], label="Difference")


    plt.xlabel('time (s)')
    plt.ylabel('Amplitude (V)')
    plt.title(f'Before and After {name}')

    event_labeled=False
    if events is not None and event_id is not None:
        for event in events.copy():
            event[0] -= first_samp
            if event[0] > start and event[0] < start+length and event[2] == event_id:
                if not event_labeled:
                    plt.axvline((event[0])/fs+event_onset, color='r', label='event')
                    event_labeled=True
                else:
                    plt.axvline((event[0])/fs+event_onset, color='r')
    plt.legend()
    if save_pth is not None:
        plt.savefig(save_pth)
        plt.close()
    else:
        plt.show()


def pcs_plot(pcs, target_fdr, ch_list, ch_names, info, win_list=None, figsize=(20, 9), strict=True, resolution=0.05, psd_lim=(0,50)):
    """
    Print the PCA components in a human-readable format.
    3 pcs are printed together in one image, with the first one being the mean.
    pcs: numpy array of shape (len(win)-1+#windows, #channels, len(epoch), #pc) or (#channels, len(epoch), #pc)
    target_fdr: the folder to save the images
    ch_list: the list of NAME of channels to plot
    ch_names: the list of names of ALL channels
    info: the info of the raw data
    strict: if True, assert the number of channels in pcs and ch_names should be the same. else, only a warning would be printed if pcs.shape[-3] < len(ch_names)
    resolution: the resolution for psd plot, in Hz/bin. Default is 0.05, which means 0.05Hz per bin in the psd plot. If the psd calculation fails due to too high resolution, the function will automatically reduce the resolution by half until it succeeds.
    """

    fs = info['sfreq']
    bad_chs = info['bads']
    ch_list = [ch for ch in ch_list if (ch not in bad_chs) and (ch in ch_names)]
    ch_names = [ch for ch in ch_names if ch not in bad_chs]

    if len(ch_names) != pcs.shape[-3]:
        if strict:
            raise ValueError(f"Number of channels in pcs ({pcs.shape[-3]}) does not match number of channels in ch_names ({len(ch_names)})")
        elif len(ch_names) > pcs.shape[-3]:
            print(f"WARNING: number of ch_names is longer then pcs.shape. the last {len(ch_names)-pcs.shape[-3]} channels would be ignored.")
            ch_names = ch_names[:pcs.shape[-3]]
        else:
            raise ValueError(f"Number of channels in pcs ({pcs.shape[-3]}) is larger than number of channels in ch_names ({len(ch_names)})")

    x = np.arange(pcs.shape[-2]) / fs
    n_fft = int(np.round(fs / resolution))
    if len(pcs.shape) == 3:
        for ch_name in ch_list:
            ch_idx = ch_names.index(ch_name)

            fig, axs = plt.subplots(pcs.shape[-1], 2, figsize=figsize, squeeze=False)
            for npc in range(pcs.shape[-1]):
                axs[npc,0].plot(x, pcs[ch_idx, :, npc], label=f"PC{npc}")
                axs[npc,0].set_title(f"PC{npc} for channel {ch_name}")
                axs[npc, 0].set_xlim([x[0], x[-1]])
                axs[npc,0].legend()

                freqs, psd = welch(pcs[ch_idx, :, npc], fs, nperseg=min(n_fft,pcs.shape[-2]))
                axs[npc,1].plot(freqs, psd, label="PSD")
                axs[npc,1].set_title(f"PC{npc} PSD for channel {ch_name}")
                axs[npc, 1].set_xlim(psd_lim)
                axs[npc,1].legend()

            axs[-1,0].set_xlabel("Time (s)")
            axs[-1,1].set_xlabel("Frequency (Hz)")
            fig.tight_layout()
            fig.savefig(os.path.join(target_fdr, f"pc_{ch_name}.png"))
            plt.close(fig)
    elif len(pcs.shape) == 4:
        for ch_name in ch_list:
            ch_idx = ch_names.index(ch_name)
            for win_idx in (win_list if win_list is not None else np.random.choice(np.arange(pcs.shape[0]), 1)):
                fig, axs = plt.subplots(pcs.shape[-1], 2, figsize=figsize, squeeze=False)
                for npc in range(pcs.shape[-1]):
                    axs[npc, 0].plot(x, pcs[win_idx, ch_idx, :, npc], label=f"PC{npc}")
                    axs[npc, 0].set_title(f"PC{npc} for No. {win_idx} window of channel {ch_name}")
                    axs[npc, 0].set_xlim([x[0], x[-1]])
                    axs[npc, 0].legend()

                    freqs, psd = welch(pcs[win_idx, ch_idx, :, npc], fs, nperseg=min(n_fft, pcs.shape[-2]))
                    axs[npc,1].plot(freqs, psd , label="PSD")
                    axs[npc, 1].set_title(f"PC{npc} PSD for No. {win_idx} window of channel {ch_name}")
                    axs[npc, 1].set_xlim(psd_lim)
                    axs[npc, 1].legend()

                axs[-1,0].set_xlabel("Time (s)")
                axs[-1,1].set_xlabel("Frequency (Hz)")
                fig.tight_layout()
                fig.savefig(os.path.join(target_fdr, f"pc_{ch_name}_win_{win_idx}.png"))
                plt.close(fig)
    else:
        raise ValueError("pcs should be of shape (#ch, len(ep), #pc) or (len(win)-1+#windows, #ch, len(ep), #pc)")
