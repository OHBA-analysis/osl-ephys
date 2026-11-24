"""Preprocessing QC: the plots and numbers of a session, saved for the report.

:func:`save_qc_plots` writes ``summary.json`` and the WebP plots to
``qc_dir/preproc/<id>/``, where :func:`osl_ephys.report.generate_report`
looks for them. Each ``plot_*`` function draws one of those plots and can be
used on its own. The other sections save their QC the same way:
:func:`osl_ephys.parcellation.plotting.save_qc_plots` and the surface and
coregistration plots of :mod:`osl_ephys.rhino`.
"""

from __future__ import annotations

import logging
import warnings
from copy import deepcopy
from pathlib import Path

import mne
import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mne.channels.channels import channel_type
from scipy.ndimage import uniform_filter1d

from osl_ephys.report import headless, save_figure, save_summary

logger = logging.getLogger("osl_ephys")


def _channel_types(raw, exclude="bads"):
    """The channel groups shown separately, with their picks (empty ones dropped).

    CTF axial gradiometers are typed as magnetometers by MNE and their
    reference gradiometers as gradiometers, so they are named accordingly.
    """
    if raw.info["dev_ctf_t"] is not None:
        groups = {
            "Axial gradiometers": mne.pick_types(
                raw.info, meg="mag", ref_meg=False, exclude=exclude
            ),
            "Reference gradiometers": mne.pick_types(
                raw.info, meg="grad", exclude=exclude
            ),
        }
    else:
        groups = {
            "Magnetometers": mne.pick_types(raw.info, meg="mag", exclude=exclude),
            "Gradiometers": mne.pick_types(raw.info, meg="grad", exclude=exclude),
        }
    groups["EEG"] = mne.pick_types(raw.info, eeg=True, exclude=exclude)
    groups["CSD"] = mne.pick_types(raw.info, csd=True, exclude=exclude)
    return {name: picks for name, picks in groups.items() if len(picks) > 0}


def _bad_segment_spans(raw):
    """The (start, end) times of the bad segment annotations."""
    return [
        (a["onset"] - raw.first_time, a["onset"] + a["duration"] - raw.first_time)
        for a in raw.annotations
        if "bad_segment" in a["description"]
    ]


def plot_flowchart(raw, filename=None):
    """The preprocessing steps recorded in the file, as a flowchart per run."""
    from osl_ephys.preproc.dataset import get_config_from_fif, plot_preproc_flowchart

    if not raw.info.get("description"):
        return None
    configs = get_config_from_fif(raw)
    if len(configs) == 0:
        return None
    fig, axes = plt.subplots(
        nrows=len(configs), ncols=1, figsize=(12, 6 * len(configs))
    )
    axes = np.atleast_1d(axes)
    for i, (config, ax) in enumerate(zip(configs, axes)):
        title = f"Preprocessing run {i + 1}" if len(configs) > 1 else None
        plot_preproc_flowchart(config, fig=fig, ax=ax, title=title)
    if filename is not None:
        save_figure(fig, filename)
    return fig


def plot_raw_data(raw, filename=None):
    """Every M/EEG channel over the whole recording, as MNE's browser shows it."""
    mne.viz.set_browser_backend("matplotlib")
    fig = (
        raw.copy()
        .pick(["meg", "eeg"])
        .plot(n_channels=np.inf, duration=raw.times[-1], use_opengl=False, show=False)
    )
    if filename is not None:
        save_figure(fig, filename)
    return fig


def plot_sum_square(raw, filename=None, exclude_bads=False):
    """The sum of squares across channels over time, per channel type.

    Bad segments are shaded; with ``exclude_bads`` the bad channels are left
    out and the bad segments are flattened to the mean of the good data.
    """
    groups = _channel_types(raw, exclude="bads" if exclude_bads else [])
    if not groups:
        return None
    t = raw.times
    x = raw.get_data()
    spans = _bad_segment_spans(raw)

    fig, axes = plt.subplots(nrows=len(groups), ncols=1, figsize=(16, 4))
    axes = np.atleast_1d(axes)
    for ax, (name, picks) in zip(axes, groups.items()):
        ss = np.sum(x[picks] ** 2, axis=0)
        if exclude_bads:
            good = raw.get_data(picks=picks, reject_by_annotation="NaN")
            fill = np.mean(ss[~np.isnan(good[0])])
            for start, end in spans:
                ss[(t >= start) & (t <= end)] = fill
        ss = uniform_filter1d(ss, int(raw.info["sfreq"]))
        ax.plot(t, ss)
        ax.legend([name], frameon=False, fontsize=16)
        ax.set_xlim(t[0], t[-1])
        for start, end in spans:
            ax.axvspan(start, end, color="red", alpha=0.8)
    axes[0].set_title("Sum-square across channels")
    axes[-1].set_xlabel("Time (seconds)")
    fig.tight_layout()
    if filename is not None:
        save_figure(fig, filename)
    return fig


def plot_channel_stds(raw, filename=None, exclude_bad_segments=True):
    """The distribution of the channels' standard deviations, per channel type.

    The bad channels are marked with dashed lines.
    """
    groups = _channel_types(raw, exclude=[])
    if not groups:
        return None
    x = raw.get_data(reject_by_annotation="omit" if exclude_bad_segments else None)
    bad = [raw.ch_names.index(ch) for ch in raw.info["bads"]]

    fig, axes = plt.subplots(nrows=1, ncols=len(groups), figsize=(9, 3.5))
    axes = np.atleast_1d(axes)
    for ax, (name, picks) in zip(axes, groups.items()):
        stds = x[picks].std(axis=1)
        ax.hist(stds, bins=24, histtype="step")
        for i in np.intersect1d(picks, bad):
            ax.axvline(x[i].std(), linestyle="--", color="tab:red")
        ax.set_xlabel("Standard deviation")
        ax.set_ylabel("Channel count")
        ax.set_title(name)
    fig.tight_layout()
    if filename is not None:
        save_figure(fig, filename)
    return fig


def plot_channel_dists(raw, filename=None, exclude_bads=True):
    """The distribution of the channels' standard deviations (see plot_channel_stds)."""
    return plot_channel_stds(raw, filename, exclude_bad_segments=exclude_bads)


def plot_sensors(raw, filename=None):
    """The sensor positions with the bad channels highlighted.

    Neuromag 306 systems have three sensors at each position, so each coil
    type is drawn separately.
    """
    if raw.get_montage() is None:
        return None
    coil_types = {c["coil_type"] for c in raw.info["chs"]}
    if 3012 in coil_types:
        with open(Path(__file__).parents[1] / "files" / "neuromag306_info.yml") as f:
            channels = yaml.safe_load(f)
        names = ["grad_longitude", "grad_lattitude"]
        if 3024 in coil_types:
            names = ["mag"] + names
        fig, axes = plt.subplots(1, len(names), figsize=(16, 4))
        for ax, name in zip(axes, names):
            raw.copy().pick(channels[name]).plot_sensors(
                axes=ax, show=False, verbose="ERROR"
            )
            ax.set_title(name.replace("_", " "))
        fig.tight_layout()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].axis("off")
        axes[2].axis("off")
        raw.plot_sensors(show=False, axes=axes[1])
    if filename is not None:
        save_figure(fig, filename)
    return fig


def plot_spectra(raw, filename=None, fmin=None, fmax=None):
    """The power spectrum of each channel, per channel type."""
    groups = _channel_types(raw)
    groups.pop("CSD", None)
    groups.pop("Reference gradiometers", None)
    if not groups:
        return None
    fig, axes = plt.subplots(nrows=len(groups), ncols=1, figsize=(8, 7))
    axes = np.atleast_1d(axes)
    kwargs = {}
    if fmin is not None:
        kwargs["fmin"] = fmin
    if fmax is not None:
        kwargs["fmax"] = min(fmax, raw.info["sfreq"] / 2)
    for ax, (name, picks) in zip(axes, sorted(groups.items())):
        raw.compute_psd(
            picks=picks, n_fft=int(raw.info["sfreq"] * 2), verbose=0, **kwargs
        ).plot(axes=ax, show=False)
        ax.set_title(name, fontsize=12)
    if filename is not None:
        save_figure(fig, filename)
    return fig


def plot_frequency_bands(raw, filename=None):
    """Topographies of the power in the standard frequency bands, per channel type."""
    from osl_ephys.parcellation.plotting import plot_source_topo

    is_parc = any("parcel" in ch for ch in raw.ch_names)
    if is_parc:
        groups = {"Parcels": mne.pick_types(raw.info, misc=True, exclude="bads")}
    else:
        groups = _channel_types(raw)
        groups.pop("CSD", None)
        # The CTF reference sensors share positions, so cannot be a topography
        groups.pop("Reference gradiometers", None)
    # Only the channel types with positions can be drawn as topographies
    for name, picks in list(groups.items()):
        try:
            if raw.copy().pick(picks).get_montage() is None:
                del groups[name]
        except Exception:
            del groups[name]
    if not groups:
        return None

    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 80), (80, 150)]
    names = ["Delta", "Theta", "Alpha", "Beta", "Gamma", "High gamma"]
    keep = [
        i
        for i, (f1, f2) in enumerate(bands)
        if f2 > raw.info["highpass"] and f1 < raw.info["lowpass"]
    ]
    bands = [bands[i] for i in keep]
    names = [names[i] for i in keep]

    fig, axes = plt.subplots(
        nrows=len(groups), ncols=len(bands), figsize=(8, 0.5 + 1.5 * len(groups))
    )
    axes = np.array(axes).reshape(len(groups), len(bands))
    for row, (name, picks) in enumerate(sorted(groups.items())):
        zscored = deepcopy(raw).apply_function(
            lambda x: (x - np.mean(x)) / np.std(x), picks=picks
        )
        psd = zscored.compute_psd(
            picks=picks, n_fft=int(raw.info["sfreq"] * 2), verbose=0
        )
        for col, (f1, f2) in enumerate(bands):
            ax = axes[row, col]
            inds = (psd.freqs > f1) & (psd.freqs <= f2)
            p = psd.get_data()[:, inds].mean(axis=1)
            if is_parc:
                plot_source_topo(p / p.std(), axis=ax, cmap="hot")
            else:
                mne.viz.plot_topomap(p, psd.info, axes=ax, cmap="hot")
            if row == 0:
                ax.set_title(f"{names[col]}\n({f1}-{f2} Hz)")
            if col == 0:
                ax.text(
                    -0.1,
                    0.5,
                    name,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10,
                )
    if filename is not None:
        save_figure(fig, filename)
    return fig


def plot_digitisation(raw, filename=None):
    """The digitised fiducials, HPI coils, EEG electrodes and headshape points."""
    if not raw.info["dig"]:
        return None
    styles = {1: "r^", 2: "mo", 3: "g+", 4: "b."}
    views = [("Top view", 0, 1), ("Front view", 0, 2), ("Side view", 1, 2)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, i, j) in zip(axes, views):
        ax.set_aspect("equal")
        for point in raw.info["dig"]:
            if point["kind"] in styles:
                ax.plot(point["r"][i], point["r"][j], styles[point["kind"]])
        ax.set_title(title)
    axes[3].axis("off")
    legend = [
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Fiducial",
            markerfacecolor="r",
            markersize=14,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="HPI",
            markerfacecolor="m",
            markersize=14,
        ),
        Line2D(
            [0],
            [0],
            marker="+",
            color="w",
            label="EEG",
            markeredgecolor="g",
            markersize=14,
        ),
        Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            label="Headshape",
            markerfacecolor="b",
            markersize=14,
        ),
    ]
    axes[3].legend(handles=legend, loc="center", frameon=False)
    fig.tight_layout()
    if filename is not None:
        save_figure(fig, filename)
    return fig


def head_movement(raw):
    """The head position over the recording from the continuous HPI, if recorded.

    Returns
    -------
    head_pos : np.ndarray or None
        MNE's head position array, or None if the file has no cHPI.
    """
    try:
        _, hpi_pick, _ = mne.chpi.get_chpi_info(raw.info, on_missing="ignore")
    except Exception:
        return None
    if hpi_pick is None:
        return None
    try:
        amplitudes = mne.chpi.compute_chpi_amplitudes(
            raw, t_step_min=1.0, verbose=False
        )
        locs = mne.chpi.compute_chpi_locs(raw.info, amplitudes, verbose=False)
        return mne.chpi.compute_head_pos(raw.info, locs, verbose=False)
    except Exception as e:
        logger.warning(f"Head movement not computed: {e}")
        return None


def plot_head_movement(raw, filename=None, head_pos=None):
    """The head position traces from the continuous HPI, if recorded."""
    if head_pos is None:
        head_pos = head_movement(raw)
    if head_pos is None or len(head_pos) == 0:
        return None
    fig = mne.viz.plot_head_positions(head_pos, mode="traces", show=False)
    if filename is not None:
        save_figure(fig, filename)
    return fig


def _plot_channels(raw, picks, filename):
    t = raw.times
    fig, ax = plt.subplots(figsize=(16, 2))
    ax.plot(t, raw.get_data(picks).T)
    ax.set_xlim([t[0], t[-1]])
    fig.tight_layout()
    if filename is not None:
        save_figure(fig, filename)
    return fig


def plot_eog(raw, filename=None):
    """The EOG channels over the recording."""
    picks = mne.pick_types(raw.info, eog=True)
    return _plot_channels(raw, picks, filename) if len(picks) > 0 else None


def plot_ecg(raw, filename=None):
    """The ECG channels over the recording."""
    picks = mne.pick_types(raw.info, ecg=True)
    return _plot_channels(raw, picks, filename) if len(picks) > 0 else None


def plot_bad_ica(raw, ica, filename=None):
    """The properties of the rejected ICA components, one row each.

    Each row shows MNE's component properties (topography, epochs image, ERP,
    spectrum and variance), with a second topography for the gradiometers when
    the data has both magnetometers and gradiometers. The title carries the
    component's label, from either the manual labelling or MNE's automatic
    EOG/ECG labelling.
    """
    mne.set_log_level("ERROR")
    ica = deepcopy(ica)
    excluded = np.sort(np.unique(ica.exclude))[::-1]
    n = len(excluded)
    if len(ica.labels_) != n:
        ica.labels_ = {ica._ica_names[i]: "" for i in excluded}
    both = "mag" in raw.get_channel_types() and "grad" in raw.get_channel_types()

    fig = plt.figure(figsize=(16, 5 * max(n, 1)), facecolor=[0.95] * 3)
    if n == 0:
        ax = plt.subplot(111, frameon=False)
        ax.text(0.5, 0.5, "No components rejected", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    for i, component in enumerate(excluded):
        lower = 0.1 + i / (n * 1.1)
        scale = n * 1.3
        # Adapted from mne/viz/ica._create_properties_layout
        if both:
            topomap1 = [
                0.08,
                lower + 0.5 / scale,
                (0.3 - 0.08) / 2 + 0.08,
                0.45 / scale,
            ]
            topomap2 = [(0.3 - 0.08) / 2 + 0.08, lower + 0.5 / scale, 0.3, 0.45 / scale]
        else:
            topomap1 = [0.08, lower + 0.5 / scale, 0.3, 0.45 / scale]
        layout = [
            ("topomap", topomap1),
            ("image", [0.5, lower + 0.6 / scale, 0.45, 0.35 / scale]),
            ("erp", [0.5, lower + 0.5 / scale, 0.45, 0.1 / scale]),
            ("spectrum", [0.08, lower + 0.1 / scale, 0.32, 0.3 / scale]),
            ("variance", [0.5, lower + 0.025 / scale, 0.45, 0.25 / scale]),
        ]
        axes = [fig.add_axes(loc, label=name) for name, loc in layout]
        ica.plot_properties(raw, picks=component, axes=axes, show=False, verbose=0)

        name = ica._ica_names[component]
        if name in ica.labels_:
            # The manual labelling convention
            label = str(ica.labels_[name]).upper()
        else:
            # MNE's automatic labelling convention
            flags = [
                k.upper() for k in ("eog", "ecg") if component in ica.labels_.get(k, [])
            ]
            label = "/".join(flags)
        title = f"{name} - {label}" if label else name
        if both:
            ax2 = fig.add_axes(topomap2, label="topomap2")
            mne.viz.ica._plot_ica_topomap(
                ica, component, ch_type="grad", axes=ax2, show=False
            )
            ax2.set_title(f"{title}\n (grad)")
            title = f"{title}\n (mag)"
        axes[0].set_title(title, fontsize=12)

        # Whether the component has already been removed from the data
        source = ica._transform_raw(raw, 0, len(raw.times))[component]
        removed = np.isclose(source.mean(), 0, atol=1e-9)
        fig.text(
            0.5,
            lower + 1 / scale,
            (
                "Excluded from preprocessed data"
                if removed
                else "Not yet excluded from data"
            ),
            ha="center",
            va="center",
            fontsize=14,
            color="black" if removed else "red",
        )
    if filename is not None:
        save_figure(fig, filename)
    return fig


def plot_evoked(epochs, filename=None):
    """The evoked response of each condition, as the global field power per
    channel type.

    Up to eight conditions are drawn as separate lines; with more, every
    epoch is averaged together.
    """
    if epochs is None or len(epochs) == 0:
        return None
    types = [t for t in ("mag", "grad", "eeg") if t in epochs.get_channel_types()]
    if not types:
        return None
    conditions = list(epochs.event_id)
    if 1 < len(conditions) <= 8:
        evokeds = {c: epochs[c].average() for c in conditions if len(epochs[c]) > 0}
    else:
        evokeds = {"all": epochs.average()}
    fig, axes = plt.subplots(1, len(types), figsize=(6 * len(types), 4))
    for ax, ch_type in zip(np.atleast_1d(axes), types):
        mne.viz.plot_compare_evokeds(
            evokeds,
            picks=ch_type,
            combine="gfp",
            axes=ax,
            show=False,
            show_sensors=False,
            legend="upper right" if len(evokeds) > 1 else False,
        )
        ax.set_title(ch_type)
    fig.tight_layout()
    if filename is not None:
        save_figure(fig, filename)
    return fig


def plot_events(events, event_id, filename=None):
    """The events over the recording, by event code."""
    if events is None or len(events) == 0:
        return None
    fig = mne.viz.plot_events(events, event_id=event_id, show=False)
    if filename is not None:
        save_figure(fig, filename)
    return fig


def _count_events(raw):
    """The number of EOG and ECG events, when the channels to find them exist."""
    counts = {}
    if len(mne.pick_types(raw.info, eog=True)) > 0:
        try:
            counts["n_eog_events"] = len(
                mne.preprocessing.find_eog_events(raw, verbose=False)
            )
        except Exception as e:
            logger.warning(f"EOG events not counted: {e}")
    if len(mne.pick_types(raw.info, ecg=True)) > 0:
        try:
            events, _, _ = mne.preprocessing.find_ecg_events(raw, verbose=False)
            counts["n_ecg_events"] = len(events)
        except Exception as e:
            logger.warning(f"ECG events not counted: {e}")
    return counts


def scan_summary(raw):
    """The numbers describing a recording, for ``summary.json``."""
    summary = {
        "n_samples": int(raw.n_times),
        "sfreq": float(raw.info["sfreq"]),
        "total_duration_s": round(float(raw.times[-1]), 1),
        "n_channels": int(raw.info["nchan"]),
    }
    types, counts = np.unique(
        [channel_type(raw.info, c) for c in range(raw.info["nchan"])],
        return_counts=True,
    )
    summary["channel_types"] = {t: int(c) for t, c in zip(types, counts)}
    if raw.info["dig"]:
        names = {1: "fiducials", 2: "hpi", 3: "eeg", 4: "headshape"}
        kinds = [d["kind"] for d in raw.info["dig"]]
        summary["digitisation"] = {
            name: int(sum(k == kind for k in kinds)) for kind, name in names.items()
        }
    try:
        summary["n_hpi_coils"] = len(raw.info["hpi_meas"][0]["hpi_coils"])
    except (KeyError, IndexError, TypeError):
        pass
    if len(mne.pick_types(raw.info, meg=False, ref_meg=False, stim=True)) > 0:
        events = mne.find_events(raw, min_duration=5 / raw.info["sfreq"], verbose=False)
        codes, counts = np.unique(events[:, 2], return_counts=True)
        summary["events"] = {str(c): int(n) for c, n in zip(codes, counts)}

    # Bad segments, overall and per channel type (the annotations name it)
    bad = np.zeros(raw.n_times, dtype=bool)
    per_type = {}
    for a in raw.annotations:
        if not a["description"].startswith("bad"):
            continue
        start = a["onset"] - raw.first_time
        inds = (raw.times >= start) & (raw.times <= start + a["duration"])
        bad |= inds
        for modality in ("mag", "grad", "eeg"):
            if modality in a["description"]:
                per_type.setdefault(modality, np.zeros(raw.n_times, dtype=bool))
                per_type[modality] |= inds
    summary["bad_duration_s"] = round(float(bad.sum() / raw.info["sfreq"]), 1)
    summary["bad_percent"] = round(float(100 * bad.mean()), 1)
    for modality, inds in per_type.items():
        summary[f"bad_percent_{modality}"] = round(float(100 * inds.mean()), 1)
    # Interpolated channels are kept in info["temp"] by the interpolate_bads step
    bads = raw.info["bads"] or (raw.info.get("temp") or {}).get("bads", [])
    summary["bad_channels"] = list(bads)
    summary["n_bad_channels"] = len(bads)
    return summary


def print_scan_summary(raw):
    """Print a text summary of a recording."""
    print(f"Datafile: {raw.filenames}")
    print(f"Project name: {raw.info['proj_name']}")
    print(f"Experimenter: {raw.info['experimenter']}")
    print(f"Data acquired: {raw.info['meas_date']}")
    summary = scan_summary(raw)
    print(
        f"{summary['n_samples']} samples at {summary['sfreq']} Hz - "
        f"{summary['total_duration_s']} seconds"
    )
    if "n_hpi_coils" in summary:
        print(f"{summary['n_hpi_coils']} HPI coils acquired")
    print(f"{summary['n_channels']} channels acquired")
    for name, count in summary["channel_types"].items():
        print(f"\t{name}: {count}")
    if "digitisation" in summary:
        print("Digitisation points")
        for name, count in summary["digitisation"].items():
            print(f"\t{name}: {count}")
    if "events" in summary:
        print("Events")
        for code, count in summary["events"].items():
            print(f"\t{code}: {count}")
    print(
        f"Bad segments: {summary['bad_duration_s']}/{summary['total_duration_s']} "
        f"seconds ({summary['bad_percent']}%)"
    )
    print(f"Bad channels: {summary['bad_channels'] or 'none'}")


def save_qc_plots(
    qc_dir,
    id,
    raw,
    ica=None,
    events=None,
    event_id=None,
    epochs=None,
    custom_figures=None,
    plots="all",
):
    """Save the preprocessing QC of a session for the report.

    Writes to ``qc_dir/preproc/<id>/``: ``summary.json`` (the numbers of
    :func:`scan_summary`, the EOG/ECG event counts, the head movement and the
    ICA rejections) and the plots of every ``plot_*`` function in this module
    as WebP files, plus ``custom_<name>.webp`` for each custom figure.

    Parameters
    ----------
    qc_dir : str or Path
        The QC directory.
    id : str
        The session id.
    raw : mne.io.Raw
        The preprocessed data.
    ica : mne.preprocessing.ICA or list of them, optional
        The fitted ICA, for the rejected components plot. A session that ran
        ICA several times (e.g. on the MEG and then the EEG channels) passes
        them all: the plots are ``ica.webp``, ``ica_2.webp``, ...
    events, event_id : optional
        The events and their ids, for the events plot.
    epochs : mne.Epochs, optional
        The epoched data, for the evoked responses plot and the epoch counts.
    custom_figures : dict, optional
        Matplotlib figures by name, saved as ``custom_<name>.webp``.
    plots : "all" or list of str, optional
        The plots to save, by file name without the extension (``"psd"``,
        ``"sum_square"``, ...). ``"all"`` saves every plot; a short list is
        worth it for large datasets, where few plots are looked at.

    Returns
    -------
    summary : dict
        What was written to ``summary.json``.
    """
    output_dir = Path(qc_dir) / "preproc" / id
    output_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", message="nperseg.*is greater than input length.*")
    warnings.filterwarnings(
        "ignore", message=".*more than 20 mm from head frame origin.*"
    )
    with headless():
        return _save_qc_plots(
            output_dir, raw, ica, events, event_id, epochs, custom_figures, plots
        )


def _save_qc_plots(
    output_dir, raw, ica, events, event_id, epochs, custom_figures, plots
):
    def wanted(name):
        return plots == "all" or name in plots

    summary = scan_summary(raw)
    summary.update(_count_events(raw))
    icas = (
        [] if ica is None else (list(ica) if isinstance(ica, (list, tuple)) else [ica])
    )
    if icas:
        summary["ica_n_components"] = [int(i.n_components_) for i in icas]
        summary["ica_n_excluded"] = sum(len(i.exclude) for i in icas)
        summary["ica_excluded_labels"] = [
            {
                str(i._ica_names[c]): str(i.labels_.get(i._ica_names[c], ""))
                for c in i.exclude
            }
            for i in icas
        ]
    if epochs is not None:
        summary["n_epochs"] = len(epochs)
        summary["n_epochs_dropped"] = int(sum(len(log) > 0 for log in epochs.drop_log))
    head_pos = head_movement(raw) if wanted("head_movement") else None
    if head_pos is not None and len(head_pos) > 0:
        # Largest distance from the initial position, in mm
        translation = head_pos[:, 4:7] - head_pos[0, 4:7]
        summary["head_movement_mm"] = round(
            float(np.max(np.linalg.norm(translation, axis=1)) * 1000), 1
        )
    save_summary(output_dir, summary)

    plotters = {
        "flowchart": lambda f: plot_flowchart(raw, f),
        "sum_square": lambda f: plot_sum_square(raw, f),
        "sum_square_exclude_bads": lambda f: plot_sum_square(raw, f, exclude_bads=True),
        "channel_stds": lambda f: plot_channel_stds(raw, f),
        "sensors": lambda f: plot_sensors(raw, f),
        "psd": lambda f: plot_spectra(raw, f, fmin=1, fmax=48),
        "psd_full": lambda f: plot_spectra(raw, f),
        "frequency_bands": lambda f: plot_frequency_bands(raw, f),
        "digitisation": lambda f: plot_digitisation(raw, f),
        "eog": lambda f: plot_eog(raw, f),
        "ecg": lambda f: plot_ecg(raw, f),
        "head_movement": lambda f: plot_head_movement(raw, f, head_pos=head_pos),
        "events": lambda f: plot_events(events, event_id, f),
        "evoked": lambda f: plot_evoked(epochs, f),
        "raw_data": lambda f: plot_raw_data(raw, f),
    }
    for n, this_ica in enumerate(icas, start=1):
        name = "ica" if n == 1 else f"ica_{n}"
        plotters[name] = lambda f, i=this_ica: plot_bad_ica(raw, i, f)
    for name, plotter in plotters.items():
        if not wanted(name):
            continue
        try:
            plotter(output_dir / f"{name}.webp")
        except Exception as e:
            logger.warning(f"QC plot {name} failed: {e}")
        plt.close("all")

    for name, fig in (custom_figures or {}).items():
        if isinstance(fig, plt.Figure):
            save_figure(fig, output_dir / f"custom_{name}.webp")
        else:
            logger.warning(f"Custom figure {name} is not a matplotlib figure; skipped.")
    return summary


def gen_report_from_fif(infiles, qc_dir, outdir=None, ftype=None, plots="all"):
    """Regenerate the QC of saved preprocessed files and the report.

    Parameters
    ----------
    infiles : str or list of str
        The preprocessed fif files (their ICA and events files are read from
        alongside them).
    qc_dir : str or Path
        The QC directory to write to.
    outdir : str or Path, optional
        The output directory, for the surfaces and coregistration QC.
    ftype : str, optional
        The fif files' suffix (``preproc-raw`` by default).
    plots : "all" or list of str, optional
        Passed to :func:`save_qc_plots`.

    Returns
    -------
    path : Path
        The report.
    """
    from osl_ephys.preproc.dataset import read_dataset
    from osl_ephys.report import generate_report

    if isinstance(infiles, (str, Path)):
        infiles = [infiles]
    suffix = f"_{ftype or 'preproc-raw'}"
    sessions = []
    for infile in infiles:
        dataset = read_dataset(str(infile), preload=True, ftype=ftype)
        # The session id is the file name without the preprocessing suffix
        id_ = Path(infile).stem
        if id_.endswith(suffix):
            id_ = id_[: -len(suffix)]
        logger.info(f"QC plots for {id_}")
        save_qc_plots(
            qc_dir,
            id_,
            dataset["raw"],
            ica=dataset.get("ica"),
            events=dataset.get("events"),
            event_id=dataset.get("event_id"),
            epochs=dataset.get("epochs"),
            plots=plots,
        )
        sessions.append({"id": id_})
    return generate_report(qc_dir, sessions, outdir)


def plot_artefact_scan(raw, output_dir):
    """The evoked responses of OHBA's artefact scan, one plot per artefact.

    The scan's events are eye movements (code 1), blinks (5), swallows (6),
    breaths (7), shrugs (8), clenches (9) and button presses (257). The EOG
    traces and the joint plots of each MEG channel type are saved as
    ``<artefact>_<type>.webp`` in ``output_dir``.
    """
    output_dir = Path(output_dir)
    events = mne.find_events(raw, min_duration=2 / raw.info["sfreq"])
    modalities = ["mag", "grad"]

    def joint_plots(name, epochs):
        evoked = epochs.average()
        for m in modalities:
            fig = evoked.plot_joint(show=False, picks=m)
            save_figure(fig, output_dir / f"{name}_{m}.webp")

    # Eye movements, with the fixation and movement periods marked
    epochs = mne.Epochs(
        raw, events, event_id={"moves": 1}, tmin=-1, tmax=16.5, preload=True
    )
    x = epochs.get_data()
    yl = np.abs(x[:, :2, :]).max() * 1.1
    fig = plt.figure(figsize=(16, 5))
    for i in range(2):
        ax = plt.subplot(1, 2, i + 1)
        for start in range(-1, 15, 1):
            colour = "r" if start in (0, 4, 8, 12) else "grey"
            width = 1 if start == -1 else 2
            if start in (-1, 0, 2, 4, 6, 8, 10, 12, 14):
                ax.add_artist(
                    Rectangle((start, -yl), width, yl * 2, alpha=0.2, facecolor=colour)
                )
        for t, label in [
            (-0.5, "fix"),
            (1, "Up"),
            (5, "Right"),
            (9, "Down"),
            (13, "Left"),
        ]:
            ax.text(t, yl, label, ha="center")
        ax.plot(epochs.times, x[:, i, :].T)
        ax.set_ylim(-yl, yl * 1.1)
    save_figure(fig, output_dir / "eyemove_eog.webp")
    joint_plots("eyemove", epochs)

    # Blinks
    epochs = mne.Epochs(
        raw, events, event_id={"blinks": 5}, tmin=-0.2, tmax=2, preload=True
    )
    x = epochs.get_data()
    fig = plt.figure(figsize=(16, 4))
    for i in range(2):
        ax = plt.subplot(1, 2, i + 1)
        ax.plot(epochs.times, x[:, i, :].T)
        ax.set_title(raw.ch_names[i])
    save_figure(fig, output_dir / "blink_eog.webp")
    joint_plots("blink", epochs)

    for name, code, tmin, tmax in [
        ("swallow", 6, -0.2, 2),
        ("breathe", 7, -0.5, 5),
        ("shrug", 8, -0.5, 3),
        ("clench", 9, -0.5, 3),
        ("buttonpress", 257, -0.5, 2),
    ]:
        if not np.any(events[:, 2] == code):
            continue
        epochs = mne.Epochs(
            raw, events, event_id={name: code}, tmin=tmin, tmax=tmax, preload=True
        )
        joint_plots(name, epochs)
    plt.close("all")
