"""All matplotlib rendering primitives for manual_ica.

Every function here writes one or more SVGs to disk. The orchestrator
(``ica.manual_ica``) is the only caller; if you need a *new* per-IC plot,
add it here and call it from manual_ica.

Layout convention: the per-IC main figure is 12 x 5 inches; the per-IC
zoom strips are 12 x 1; stacked, the HTML page renders them as a 12 x 6
(2:1) viewer.
"""
import math
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import mne
from ...utils.logger import log_or_print

from .helpers import _good_mask


# ── axis positions for the IC summary figure (figure-fraction coords) ──────
# Main figure is 12 x 5; zoom figure is 12 x 1. Inside the main:
#   row 1 ~ 40% of fig height (topo + img/ERP)
#   row 2 ~ 28% (spectrum + variance)
#   row 3 ~ 14% (full timecourse)
# the remainder is gaps/labels. Topomap is square: 0.167 * 12in = 2 in wide,
# 0.40 * 5in = 2 in tall.
# [left, bottom, width, height]
_AX_TOPO     = [0.04, 0.55, 0.167, 0.40]
_AX_IMAGE    = [0.25, 0.66, 0.71,  0.29]
_AX_ERP      = [0.25, 0.55, 0.71,  0.10]

_AX_SPEC     = [0.04, 0.27, 0.44, 0.22]
_AX_VAR      = [0.55, 0.27, 0.34, 0.22]
_AX_VHIST    = [0.90, 0.27, 0.06, 0.22]

_AX_TC_FULL  = [0.04, 0.08, 0.92, 0.10]   # row 3: full IC timecourse


# ── small numeric helpers ─────────────────────────────────────────────────

def _nice_tick_step(span, target_n=8):
    """Pick a nice major-tick step (1, 2, 5 x 10^k) for a given axis span."""
    if span <= 0:
        return 1.0
    target = span / target_n
    exp = math.floor(math.log10(target))
    base = target / (10 ** exp)
    if base < 1.5:
        nice = 1
    elif base < 3.5:
        nice = 2
    elif base < 7.5:
        nice = 5
    else:
        nice = 10
    return nice * (10 ** exp)


def _robust_minmax(arr, k=15.0):
    """Median +/- k*MAD on finite values; falls back to true min/max if MAD=0."""
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return -1.0, 1.0
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if mad == 0.0:
        return float(np.min(a)), float(np.max(a))
    return med - k * mad, med + k * mad


# ── SVG post-processing ───────────────────────────────────────────────────

_SVG_ROOT_RX = re.compile(rb'<svg\b([^>]*)>')


def _force_svg_stretch(path):
    """Add ``preserveAspectRatio="none"`` to the root ``<svg>`` element.

    Returns True if we injected the attribute, False if it was already
    present. Raises if there's no <svg> root tag at all (would be a
    matplotlib regression).
    """
    data = path.read_bytes()
    m = _SVG_ROOT_RX.search(data)
    if not m:
        raise RuntimeError(
            f'no <svg> root tag in {path}; matplotlib SVG output changed?'
        )
    attrs = m.group(1)
    if b'preserveAspectRatio' in attrs:
        # Caller (or a future matplotlib release) already set it; honour
        # their choice. Worth knowing because it means our CSS-stretch
        # assumption may no longer hold --- log so the symptom is visible.
        log_or_print(
            f'[manual_ica] {path.name} already has preserveAspectRatio; '
            f'leaving it alone (CSS stretch in between_ic.html may letterbox)',
        )
        return False
    new_tag = b'<svg' + attrs + b' preserveAspectRatio="none">'
    path.write_bytes(data[:m.start()] + new_tag + data[m.end():])
    return True


# ── per-IC main SVG (12 x 5: topo + image+ERP + spec + var + full TC) ─────

def _draw_ic_properties(ica, raw, ic_idx, fig, src_data,
                        epochs_src=None, good_mask=None, is_flagged=False,
                        seg_len=2.0, spec_type='linear', spec_freq_max=45.0,
                        psd_resolution=0.05, zoom_window=20.0,
                        ecg_event_times=None):
    """Draw the IC summary into ``fig``.

    Row 1 (top):    topomap | segment image + ERP/ERF
    Row 2 (middle): single spectrum (linear or dB, 0-spec_freq_max Hz)
                    | variance scatter + KDE histogram
    Row 3 (bottom): full IC timecourse (zoom window is in a separate file ---
                    see ``_render_ic_zoom``).

    spec_type     : 'linear' (default) | 'db'
    spec_freq_max : Hz upper limit for the spectrum (default 45)
    zoom_window   : seconds in the zoom subrow (default 20)
    ecg_event_times : 1-D array of times (s, relative to recording start) of
                      detected R-peaks; ``None`` to skip heartbeat marks.
    """
    from mne.epochs import make_fixed_length_epochs
    from mne.viz.topomap import _plot_ica_topomap
    from mne.viz.epochs import plot_epochs_image
    from mne.stats.parametric import _parametric_ci
    from scipy.stats import gaussian_kde

    topo_ax    = fig.add_axes(_AX_TOPO,    label='topomap')
    image_ax   = fig.add_axes(_AX_IMAGE,   label='image')
    erp_ax     = fig.add_axes(_AX_ERP,     label='erp')
    spec_ax    = fig.add_axes(_AX_SPEC,    label='spec')
    var_ax     = fig.add_axes(_AX_VAR,     label='var')
    vhist_ax   = fig.add_axes(_AX_VHIST,   label='vhist')
    tc_full_ax = fig.add_axes(_AX_TC_FULL, label='tc_full')

    # -- 1. Topomap ---------------------------------------------------------
    _plot_ica_topomap(ica, ic_idx, show=False, axes=topo_ax)

    # -- 2. Fixed-length epochs of ICA sources -----------------------------
    # Allow caller to pass a precomputed epochs_src/good_mask (cheaper across
    # ICs); fall back to building them here.
    if epochs_src is None:
        epochs_src = make_fixed_length_epochs(
            ica.get_sources(raw), duration=seg_len, preload=True,
            reject_by_annotation=True, proj=False, verbose=False,
        )
    if good_mask is None:
        good_mask = _good_mask(raw)
    n_epochs = len(epochs_src)
    sfreq = raw.info['sfreq']
    seg_samples = int(round(seg_len * sfreq))
    n_total_segs = max(1, raw.n_times // seg_samples)
    dropped_pct = (n_total_segs - n_epochs) / n_total_segs * 100.0
    seg_idx_orig = ((epochs_src.events[:, 0] - raw.first_samp) // seg_samples)

    # -- 3. Segment image + ERP/ERF -----------------------------------------
    plot_epochs_image(
        epochs_src, picks=ic_idx, axes=[image_ax, erp_ax],
        combine=None, colorbar=False, show=False,
        ts_args={'truncate_xaxis': False, 'show_sensors': False, 'ci': _parametric_ci},
    )
    t0, t1 = float(epochs_src.times[0]), float(epochs_src.times[-1])
    seg_step = _nice_tick_step(t1 - t0)
    image_ax.set_xlim(t0, t1)
    image_ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    image_ax.set_ylim(-0.5, n_epochs + 0.5)
    image_ax.set_title(
        f'IC {ic_idx:03d}  -  {seg_len:g}s segment image + ERP/ERF',
        fontsize=11, fontweight='bold',
        color='firebrick' if is_flagged else 'black',
    )
    image_ax.set_ylabel('Segment', fontsize=8)
    erp_ax.set_xlim(t0, t1)
    erp_ticks = np.arange(0.0, t1 + seg_step / 2, seg_step)
    erp_ax.set_xticks(erp_ticks)
    erp_ax.set_xlabel('')
    erp_ax.set_ylabel('AU', fontsize=7)
    erp_ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize=7)
    erp_ax.tick_params(axis='y', labelsize=7)

    # -- 4. PSD computation (Welch on GOOD samples only) -------------------
    n_fft = int(round(sfreq / psd_resolution))
    good_ic_data = src_data[ic_idx:ic_idx + 1, good_mask]
    n_fft = min(n_fft, good_ic_data.shape[1])
    while True:
        try:
            psds_arr, freqs = mne.time_frequency.psd_array_welch(
                good_ic_data, sfreq=sfreq,
                fmin=0, fmax=sfreq / 2.0, n_fft=n_fft, verbose=False,
            )
            break
        except (ValueError, RuntimeError) as e:
            if n_fft <= 256:
                raise
            n_fft = n_fft // 2
            log_or_print(f'[manual_ica] PSD failed ({e}), retry n_fft={n_fft}')
    psds_mean = psds_arr.squeeze()  # (n_freqs,)

    # -- 5. Single spectrum (linear by default, dB optional) ----------------
    spec_mask = freqs <= spec_freq_max
    fp, pp = freqs[spec_mask], psds_mean[spec_mask]
    if spec_type == 'db':
        pp = 10.0 * np.log10(np.maximum(pp, 1e-30))
        spec_ylabel = 'dB'
    else:
        spec_ylabel = 'Power (AU)'
    if len(fp) > 1:
        spec_ax.plot(fp, pp, color='k', lw=1)
    spec_ax.set_xlim(0, spec_freq_max)
    spec_step = _nice_tick_step(spec_freq_max)
    spec_ax.xaxis.set_major_locator(MultipleLocator(spec_step))
    spec_ax.set_xlabel(
        f'Spectrum ({spec_type})   Frequency (Hz)   range 0-{spec_freq_max:g} Hz',
        fontsize=8,
    )
    spec_ax.set_ylabel(spec_ylabel, fontsize=8)
    spec_ax.tick_params(labelsize=7)

    # -- 6. Variance scatter + KDE histogram (separate, non-overlapping) ----
    epoch_data = epochs_src.get_data()
    epoch_var = np.var(epoch_data[:, ic_idx, :], axis=1)
    var_ax.scatter(seg_idx_orig, epoch_var, alpha=0.5, facecolor=[0, 0, 0], lw=0)
    var_ax.set_xlim(-0.5, n_total_segs - 0.5)
    var_ax.set_xlabel(
        f'Segment (gaps = bad seg: {dropped_pct:.2f}% '
        f'({n_total_segs - n_epochs}/{n_total_segs}))',
        fontsize=8,
    )
    var_ax.set_ylabel('Variance (AU)', fontsize=8)
    var_ax.tick_params(labelsize=7)
    var_ylim = var_ax.get_ylim()
    vhist_ax.hist(epoch_var, orientation='horizontal', color='k', alpha=0.5)
    try:
        kde = gaussian_kde(epoch_var)
        x = np.linspace(var_ylim[0], var_ylim[1], 50)
        kde_vals = kde(x)
        kde_vals /= kde_vals.max() or 1.0
        kde_vals *= vhist_ax.get_xlim()[-1] * 0.9
        vhist_ax.plot(kde_vals, x, color='k')
    except np.linalg.LinAlgError:
        pass
    vhist_ax.set_ylim(var_ylim)
    vhist_ax.set_yticks([])
    vhist_ax.set_xlabel('')
    vhist_ax.tick_params(labelsize=6)

    # -- 7. Full IC timecourse (NaN-masked bad samples). Zoom is now its
    #       own SVG file (one per fixed time window) - see _render_ic_zoom.
    n_total   = src_data.shape[1]
    total_dur = n_total / sfreq
    f_data = src_data[ic_idx, :].astype(float, copy=True)
    f_data[~good_mask] = np.nan
    f_times = np.arange(n_total) / sfreq
    tc_full_ax.plot(f_times, f_data, lw=0.4, color='k')
    tc_full_ax.set_xlim(0, total_dur)
    f_step = _nice_tick_step(total_dur)
    tc_full_ax.xaxis.set_major_locator(MultipleLocator(f_step))
    tc_full_ax.set_xlabel(
        f'Full IC timecourse,   Time (s)   range 0-{total_dur:.0f} s',
        fontsize=8,
    )
    tc_full_ax.set_yticks([])
    for spine in ['top', 'right', 'left']:
        tc_full_ax.spines[spine].set_visible(False)
    tc_full_ax.tick_params(labelsize=7)


def _plot_ic_summary(ica, raw, ic_idx, src_data, save_dir,
                     epochs_src=None, good_mask=None,
                     is_flagged=False, latex_mode=False,
                     seg_len=2.0, spec_type='linear', spec_freq_max=45.0,
                     psd_resolution=0.05, zoom_window=20.0,
                     ecg_event_times=None):
    """Save the main per-IC SVG (rows 1+2 + full timecourse).

    The figure is saved WITHOUT ``bbox_inches='tight'`` so the SVG
    dimensions exactly match the figsize. This matters because the HTML
    page positions a click-to-jump zone + gold zoom-window overlay over
    the full-timecourse axes using the literal ``_AX_TC_FULL`` fractions;
    tight bbox would crop margins and shift those fractions.
    """
    fig = plt.figure(figsize=(12, 5))
    try:
        _draw_ic_properties(
            ica, raw, ic_idx, fig, src_data,
            epochs_src=epochs_src, good_mask=good_mask,
            is_flagged=is_flagged,
            seg_len=seg_len, spec_type=spec_type, spec_freq_max=spec_freq_max,
            psd_resolution=psd_resolution, zoom_window=zoom_window,
            ecg_event_times=ecg_event_times,
        )
    except Exception as e:
        import traceback
        log_or_print(f'[manual_ica] _draw_ic_properties failed for IC {ic_idx}: {e}')
        log_or_print(traceback.format_exc())

    stem = save_dir / f'ic_{ic_idx:03d}'
    fig.savefig(str(stem) + '.svg')
    plt.close(fig)


# ── per-IC topomap (square SVG used by the between_ic compare view) ───────

def _render_ic_topo(ica, ic_idx, save_dir):
    """Save a square topomap SVG -> ``<save_dir>/topos/ic_NNN.svg``."""
    from mne.viz.topomap import _plot_ica_topomap
    out = save_dir / 'topos' / f'ic_{ic_idx:03d}.svg'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(1.6, 1.6))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    try:
        _plot_ica_topomap(ica, ic_idx, show=False, axes=ax)
    except Exception:
        pass
    fig.savefig(str(out))
    plt.close(fig)


# ── per-IC zoom window (12x1 with axis labels, used by single_ic view) ────

def _render_ic_zoom(ic_idx, window_idx, src_data, sfreq, good_mask, save_dir,
                    zoom_window=20.0, ecg_event_times=None):
    """Render and save one zoom-window SVG for one IC.

    File: ``<save_dir>/zoomed/ic_<ic:03d>_w<window:02d>.svg``
    Window covers ``[window_idx * zoom_window, (window_idx+1) * zoom_window)``,
    clipped to the recording. Bad samples are NaN-masked so they leave gaps.
    """
    import matplotlib.transforms as mtransforms

    n_total   = src_data.shape[1]
    total_dur = n_total / sfreq
    ws = window_idx * zoom_window
    we = min(ws + zoom_window, total_dur)
    if ws >= total_dur:
        return  # past the end - skip
    i0, i1 = int(ws * sfreq), int(we * sfreq)

    z = src_data[ic_idx, i0:i1].astype(float, copy=True)
    z[~good_mask[i0:i1]] = np.nan
    t = np.arange(i0, i1) / sfreq

    # 12 x 1 figure -> stacked under the 12 x 5 main, total 12 x 6 = 2:1.
    fig = plt.figure(figsize=(12, 1))
    ax  = fig.add_axes([0.04, 0.34, 0.92, 0.62])
    ax.plot(t, z, lw=0.7, color='k')
    ax.set_xlim(ws, we)
    step = _nice_tick_step(zoom_window)
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.set_xlabel(
        f'Zoom No. {window_idx:02d}  ({ws:.0f}-{we:.0f} s)',
        fontsize=8,
    )
    ax.set_yticks([])
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis='x', labelsize=7)

    if ecg_event_times is not None:
        ax.scatter([], [], marker='^', color='red', s=22, label='R-peak (ECG)')
        if len(ecg_event_times) > 0:
            in_win = np.asarray(ecg_event_times)
            in_win = in_win[(in_win >= ws) & (in_win <= we)]
            if len(in_win) > 0:
                trans = mtransforms.blended_transform_factory(
                    ax.transData, ax.transAxes,
                )
                ax.scatter(
                    in_win, np.full(len(in_win), -0.06),
                    marker='^', color='red', s=22,
                    transform=trans, clip_on=False, zorder=5,
                )
        ax.legend(loc='upper right', fontsize=7, frameon=True, framealpha=0.85,
                  handletextpad=0.4, borderpad=0.3)

    out_dir = save_dir / 'zoomed'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_dir / f'ic_{ic_idx:03d}_w{window_idx:02d}.svg'))
    plt.close(fig)


# ── "clean" (axisless, stretched) zoom strip used by the between_ic view ──

def _render_zoom_clean(data_1d, sfreq, save_path, window_idx, zoom_window,
                       good_mask=None, robust_k=15.0, scale='robust_minmax'):
    """Single-channel "clean" zoom strip for view 2: no axes/ticks/heartbeat.

    Y-limits are robust (median +/- k*MAD over the WHOLE channel) so the
    y-scale stays stable across windows; out-of-range samples are clipped
    at the axes box and tallied as small up-/down-triangle counters in
    the right corner. Other ``scale`` values raise NotImplementedError.

    The saved SVG gets ``preserveAspectRatio="none"`` injected into its root
    element (via ``_force_svg_stretch``) so it stretches to fill whatever
    box the HTML slot has, instead of letterboxing inside it.
    """
    if scale != 'robust_minmax':
        raise NotImplementedError(f'scale={scale!r} not implemented')
    n = len(data_1d)
    total_dur = n / sfreq
    ws = window_idx * zoom_window
    we = min(ws + zoom_window, total_dur)
    if ws >= total_dur:
        return
    i0, i1 = int(ws * sfreq), int(we * sfreq)
    z = data_1d[i0:i1].astype(float, copy=True)
    if good_mask is not None:
        z[~good_mask[i0:i1]] = np.nan
    t = np.arange(i0, i1) / sfreq

    lo, hi = _robust_minmax(data_1d, k=robust_k)

    fig = plt.figure(figsize=(12, 1))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.plot(t, z, lw=0.7, color='k', clip_on=True)
    ax.set_xlim(ws, we)
    ax.set_ylim(lo, hi)
    ax.set_axis_off()

    # Clip-count indicators (NaN-safe: NaN comparisons always evaluate False).
    # Use \uXXXX escapes so the source file stays latin-1; osl-ephys's
    # append_preproc_info embeds extra_funcs source into info['description']
    # which is latin-1 only, and while it's only manual_ica that's the
    # extra_func today, keeping vis.py ASCII-clean is one less footgun.
    n_above = int(np.sum(z > hi))
    n_below = int(np.sum(z < lo))
    if n_above:
        ax.text(0.995, 0.95, f'\u25b2{n_above}', ha='right', va='top',
                transform=ax.transAxes, fontsize=8, color='#cc3333',
                fontweight='bold', clip_on=False)
    if n_below:
        ax.text(0.995, 0.05, f'\u25bc{n_below}', ha='right', va='bottom',
                transform=ax.transAxes, fontsize=8, color='#3333cc',
                fontweight='bold', clip_on=False)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path))
    plt.close(fig)
    _force_svg_stretch(save_path)
