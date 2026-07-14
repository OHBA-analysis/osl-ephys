"""Orchestration: fit ICA, score it, render diagnostics, write the HTML
review pages. The actual signal helpers / matplotlib rendering /
HTML-template substitution live in sibling modules; this file is
deliberately *just* the ``manual_ica`` extra_func that osl-ephys's
``run_proc_chain`` calls.
"""
import math

import numpy as np
import mne
from mne.preprocessing import ICA
from ...utils.logger import log_or_print

from ...utils import ensure_dir, proc_userargs
from .config  import DEFAULT_USERARGS
from .helpers import (
    _good_mask,
    _pick_supp_channels,
    _compute_slice_ratios,
    _build_scores_list,
    _resolve_subject_and_outdir,
)
from .vis    import (
    _plot_ic_summary,
    _render_ic_topo,
    _render_ic_zoom,
    _render_zoom_clean,
)
from .html   import _write_review_html


def manual_ica(dataset, userargs):
    """Fit ICA on raw EEG/MEG data and save per-IC diagnostic SVGs for
    manual review.

    ``dataset`` keys (osl-ephys ``run_proc_chain`` convention)
    ----------------------------------------------------------
    Required:
      ``raw``     : ``mne.io.Raw``  --- the data to decompose.
      ``subject`` : str             --- subject ID. Must be set by an
                                        upstream extra_func; see the README
                                        for the rationale (NB: passing
                                        ``subject`` via userargs is
                                        explicitly rejected).

    Optional:
      ``target_pth`` : str | Path
          Used as a fallback when ``userargs['outdir']`` is not given;
          IC review pages land at ``target_pth / 'ica' / subject /``.
      ``slice_interval`` : float (seconds, EEG-fMRI only)
      ``tr_interval``    : float (seconds, EEG-fMRI only)
          Together these enable an *additional* per-IC residual gradient-
          artefact (GA) score: the ratio of PSD power at slice harmonics
          (1/slice_interval and multiples) over a wider baseline window.
          Only useful for EEG-fMRI data --- if either key is absent the GA
          score is silently skipped (no ``GA: x.xx / 4`` in the scores
          bar; the ``ga_threshold`` userarg has no effect).

    Outputs written into ``dataset``:
      ``ica`` : ``mne.preprocessing.ICA``  --- the fitted decomposition.
                osl-ephys's batch.py persists this as ``<subject>_ica.fif``.

    ``userargs`` schema
    -------------------
    The full schema lives in ``osl_ephys.preprocessing.manual_ica.config.DEFAULT_USERARGS``;
    it is enforced strict --- any unknown key raises a clear KeyError.
    Highlights:

      n_components, method, max_iter, seed, l_freq, picks  --- ICA fit.
      ecg_ch, eog_ch                                       --- channel selectors
                                                               (auto-pick from
                                                               ``raw.info`` if
                                                               not given).
      ecg_threshold, eog_threshold, ga_threshold           --- auto-flag thresholds.
      seg_len, spec_type, spec_freq_max,
        psd_resolution, zoom_window, latex_mode            --- plotting knobs.
      outdir                                               --- IC review root
                                                               (per-subject
                                                               subdirs created
                                                               underneath).

    Output layout
    -------------
    Per-IC SVG (3 rows in the main 12x5 figure):
      Row 1: topomap | segment image + ERP/ERF
      Row 2: spectrum | variance scatter + KDE histogram
      Row 3: full IC timecourse

    Plus a 12x1 zoom-window SVG per IC per ``zoom_window``-s chunk, an
    axisless "clean" strip used by the between-IC compare view, a square
    topomap SVG (also for the compare view), and clean strips per
    supplementary channel (ECG / EOG / EMG). ECG / EOG / GA scores are
    embedded in the review HTML's colour-coded scores bar (the GA part
    is omitted entirely if ``slice_interval`` / ``tr_interval`` were not
    in ``dataset``).
    """
    userargs = proc_userargs(userargs, DEFAULT_USERARGS)

    if userargs['latex_mode']:
        raise NotImplementedError(
            "[manual_ica] latex_mode is temporarily disabled while the zoom "
            "panel is split out of the main SVG. Re-enable later by also "
            "rendering the per-window zooms as PDFs."
        )

    # Resolve seg_len. Default 2.0 s --- matches osl-ephys bad_segments
    # window length so a 2 s bad annotation drops at most one 2 s segment
    # instead of two adjacent half-overlapping ones (which a 2.28 s = 2*TR
    # segmentation would cause). TR-multiple alignment is available by
    # passing seg_len explicitly (e.g. seg_len=2.28).
    seg_len = userargs['seg_len']
    if seg_len is None:
        seg_len = 2.0
        log_or_print('[manual_ica] seg_len = 2.0 s (matches bad_segments window)')

    subject, ica_fdr = _resolve_subject_and_outdir(dataset, userargs)
    ensure_dir(ica_fdr)

    # -- fit ----------------------------------------------------------------
    ica = ICA(
        n_components=userargs['n_components'],
        method=userargs['method'],
        max_iter=userargs['max_iter'],
        random_state=userargs['seed'],
    )
    raw_for_fit = dataset['raw'].copy().filter(l_freq=userargs['l_freq'], h_freq=None)
    ica.fit(raw_for_fit, picks=userargs['picks'])
    del raw_for_fit
    dataset['ica'] = ica
    log_or_print(f'[manual_ica] Fitted {ica.n_components_} components')

    # -- ECG CTPS ----------------------------------------------------------
    ecg_scores = None
    ecg_idx_auto = []
    ecg_picks = mne.pick_types(dataset['raw'].info, ecg=True)
    ecg_ch = userargs['ecg_ch']
    if ecg_ch is None and len(ecg_picks) > 0:
        ecg_ch = dataset['raw'].ch_names[ecg_picks[0]]
    ecg_event_times = None  # 1-D times of detected R-peaks, for zoom-row marks
    if ecg_ch is not None:
        try:
            ecg_idx_auto, ecg_scores = ica.find_bads_ecg(dataset['raw'], ch_name=ecg_ch, method='ctps')
            log_or_print(f'[manual_ica] ECG CTPS auto-suggested ICs: {ecg_idx_auto}')
        except Exception as e:
            log_or_print(f'[manual_ica] ECG CTPS failed: {e}')
        try:
            from mne.preprocessing import find_ecg_events
            ecg_events, _, avg_pulse = find_ecg_events(
                dataset['raw'], ch_name=ecg_ch, verbose=False,
            )
            sample_rel = ecg_events[:, 0] - dataset['raw'].first_samp
            ecg_event_times = sample_rel / dataset['raw'].info['sfreq']
            log_or_print(
                f'[manual_ica] {len(ecg_event_times)} R-peaks detected '
                f'(avg {avg_pulse:.1f} bpm)'
            )
        except Exception as e:
            log_or_print(f'[manual_ica] ECG event detection failed: {e}')

    # -- EOG correlation --------------------------------------------------
    eog_scores_list = None
    eog_idx_auto = []
    eog_picks = mne.pick_types(dataset['raw'].info, eog=True)
    eog_ch = userargs['eog_ch']
    if eog_ch is None and len(eog_picks) > 0:
        eog_ch = [dataset['raw'].ch_names[i] for i in eog_picks]
    if eog_ch is not None:
        try:
            eog_idx_auto, raw_eog_scores = ica.find_bads_eog(dataset['raw'], ch_name=eog_ch)
            if isinstance(raw_eog_scores, list):
                ch_names = eog_ch if isinstance(eog_ch, list) else [eog_ch]
                eog_scores_list = list(zip(ch_names, raw_eog_scores))
            else:
                ch_name = eog_ch if isinstance(eog_ch, str) else eog_ch[0]
                eog_scores_list = [(ch_name, raw_eog_scores)]
            log_or_print(f'[manual_ica] EOG corr auto-suggested ICs: {eog_idx_auto}')
        except Exception as e:
            log_or_print(f'[manual_ica] EOG correlation failed: {e}')

    # -- IC timecourses + good-sample bookkeeping -------------------------
    sources = ica.get_sources(dataset['raw'])
    src_data = sources.get_data()
    sfreq = dataset['raw'].info['sfreq']

    good_mask = _good_mask(dataset['raw'])
    n_bad = int((~good_mask).sum())
    if n_bad > 0:
        log_or_print(
            f'[manual_ica] {n_bad}/{len(good_mask)} samples '
            f'({n_bad/len(good_mask)*100:.2f}%) inside BAD annotations - excluded'
        )
    from mne.epochs import make_fixed_length_epochs
    epochs_src = make_fixed_length_epochs(
        sources, duration=seg_len, preload=True,
        reject_by_annotation=True, proj=False, verbose=False,
    )

    # Drop ECG R-peak events landing in bad samples
    if ecg_event_times is not None and len(ecg_event_times) > 0:
        idx = np.clip((ecg_event_times * sfreq).astype(int), 0, len(good_mask) - 1)
        keep = good_mask[idx]
        n_dropped = int((~keep).sum())
        if n_dropped > 0:
            log_or_print(
                f'[manual_ica] dropped {n_dropped} R-peaks inside BAD segments'
            )
        ecg_event_times = ecg_event_times[keep]

    ecg_threshold = userargs['ecg_threshold']
    eog_threshold = userargs['eog_threshold']
    ga_threshold  = userargs['ga_threshold']

    # -- residual gradient-artefact (GA) ratio --- EEG-fMRI only ---------
    # Only computed when the caller has placed slice + TR timing into
    # `dataset` (typically by an upstream extra_func that knows the
    # acquisition is EEG-fMRI). Without these the GA score is silently
    # omitted from the per-IC scores bar in the HTML --- intentional, so
    # non-EEG-fMRI users don't see a bogus "GA: x.xx / 4" line. The info
    # log below tells EEG-fMRI users that they're missing the feature so
    # they can wire `dataset['slice_interval']` / `dataset['tr_interval']`
    # in their pipeline.
    slice_ratios = None
    if 'slice_interval' in dataset and 'tr_interval' in dataset:
        try:
            slice_ratios = _compute_slice_ratios(
                src_data, sfreq,
                dataset['slice_interval'], dataset['tr_interval'],
                good_mask=good_mask,
            )
            log_or_print(f'[manual_ica] residual GA ratios computed (max={slice_ratios.max():.2f})')
        except Exception as e:
            log_or_print(f'[manual_ica] residual GA ratios failed: {e}')
            slice_ratios = None
    else:
        log_or_print(
            "[manual_ica] residual GA scoring skipped --- not in dataset. "
            "If this is EEG-fMRI data, set dataset['slice_interval'] and "
            "dataset['tr_interval'] (seconds) in an upstream extra_func to "
            "enable per-IC GA scoring."
        )

    # Auto-flag based on USER thresholds (not MNE's defaults) so the SVG
    # title colour stays in sync with the HTML scores bar.
    flagged_ics = set()
    if ecg_scores is not None:
        flagged_ics |= {i for i in range(ica.n_components_)
                        if abs(float(ecg_scores[i])) > ecg_threshold}
    for _ch, _sc in (eog_scores_list or []):
        flagged_ics |= {i for i in range(ica.n_components_)
                        if abs(float(_sc[i])) > eog_threshold}
    if slice_ratios is not None:
        flagged_ics |= {i for i in range(ica.n_components_)
                        if float(slice_ratios[i]) > ga_threshold}

    zoom_window = userargs['zoom_window']
    n_zoom = max(1, math.ceil((src_data.shape[1] / sfreq) / zoom_window))

    supp_chs = _pick_supp_channels(dataset['raw'])
    n_supp = len(supp_chs)
    log_or_print(
        f'[manual_ica] {ica.n_components_} ICs x ({1} main + {n_zoom} zoom + '
        f'1 topo + {n_zoom} clean) SVGs + {n_supp} supp x {n_zoom} clean'
    )

    # Per-stage failure counters --- a *systematic* render bug surfaces as
    # one summary line at the end ("47/N zoom failed") rather than 47
    # near-identical tracebacks.
    fail_counts = {'main': 0, 'topo': 0, 'zoom': 0, 'clean': 0, 'supp': 0}
    n_zoom_attempts = ica.n_components_ * n_zoom
    n_supp_attempts = n_supp * n_zoom

    if n_supp:
        supp_data = dataset['raw'].get_data(picks=[d for d, _ in supp_chs])
        for i_ch, (disp, safe) in enumerate(supp_chs):
            for w in range(n_zoom):
                try:
                    _render_zoom_clean(
                        supp_data[i_ch], sfreq,
                        ica_fdr / 'zoomed_clean' / f'supp_{safe}_w{w:02d}.svg',
                        w, zoom_window, good_mask=good_mask,
                    )
                except Exception as e:
                    fail_counts['supp'] += 1
                    log_or_print(f'[manual_ica] supp zoom {disp} w{w} failed: {e}')

    for ic_idx in range(ica.n_components_):
        try:
            _plot_ic_summary(
                ica, dataset['raw'], ic_idx, src_data, ica_fdr,
                epochs_src=epochs_src, good_mask=good_mask,
                is_flagged=ic_idx in flagged_ics,
                latex_mode=False,
                seg_len=seg_len,
                spec_type=userargs['spec_type'],
                spec_freq_max=userargs['spec_freq_max'],
                psd_resolution=userargs['psd_resolution'],
                zoom_window=zoom_window,
                ecg_event_times=ecg_event_times,
            )
        except Exception as e:
            fail_counts['main'] += 1
            log_or_print(f'[manual_ica] main plot failed for IC {ic_idx}: {e}')
        try:
            _render_ic_topo(ica, ic_idx, ica_fdr)
        except Exception as e:
            fail_counts['topo'] += 1
            log_or_print(f'[manual_ica] topo IC {ic_idx} failed: {e}')
        for w in range(n_zoom):
            try:
                _render_ic_zoom(
                    ic_idx, w, src_data, sfreq, good_mask, ica_fdr,
                    zoom_window=zoom_window,
                    ecg_event_times=ecg_event_times,
                )
            except Exception as e:
                fail_counts['zoom'] += 1
                log_or_print(f'[manual_ica] zoom IC {ic_idx} w{w} failed: {e}')
            try:
                _render_zoom_clean(
                    src_data[ic_idx], sfreq,
                    ica_fdr / 'zoomed_clean' / f'ic_{ic_idx:03d}_w{w:02d}.svg',
                    w, zoom_window, good_mask=good_mask,
                )
            except Exception as e:
                fail_counts['clean'] += 1
                log_or_print(f'[manual_ica] clean zoom IC {ic_idx} w{w} failed: {e}')

    log_or_print(f'[manual_ica] Plots saved to {ica_fdr}')
    summary = (
        f'[manual_ica] render summary: '
        f"main {fail_counts['main']}/{ica.n_components_}, "
        f"topo {fail_counts['topo']}/{ica.n_components_}, "
        f"zoom {fail_counts['zoom']}/{n_zoom_attempts}, "
        f"clean {fail_counts['clean']}/{n_zoom_attempts}, "
        f"supp {fail_counts['supp']}/{n_supp_attempts} failed"
    )
    if any(fail_counts.values()):
        log_or_print('WARNING: ' + summary)
    else:
        log_or_print(summary)

    scores_list = _build_scores_list(
        ica.n_components_, ecg_scores, ecg_idx_auto, ecg_threshold,
        eog_scores_list, eog_idx_auto, eog_threshold,
        slice_ratios=slice_ratios, ga_threshold=ga_threshold,
    )
    html_path = _write_review_html(
        ica.n_components_, subject, ica_fdr,
        flagged_ics=flagged_ics, scores_list=scores_list,
        n_zoom=n_zoom,
        total_dur=src_data.shape[1] / sfreq,
        zoom_window=zoom_window,
        supp_chs=supp_chs,
    )
    log_or_print(f'[manual_ica] Review page -> {html_path}')
    return dataset


def apply_ica(dataset, userargs):
    """Apply a saved ICA solution, removing user-specified bad components.

    The in-pipeline counterpart to the ``osl-ica-apply`` CLI: pass the bad
    ICs decided during the manual browser review as ``bad_ics`` and this sets
    ``ica.exclude`` and applies the ICA to ``dataset['raw']``. Lives with the
    manual ICA pipeline (rather than the automatic semp path) because that is
    where a human hands back a list of components to remove.

    Loads / re-saves the ICA at the conventional per-subject path
    ``dataset['target_pth']/<subject>/<subject>_ica.fif`` (the same layout
    ``manual_ica`` writes), so ``dataset`` must carry ``subject`` and
    ``target_pth`` -- seed them in your project ``initialize`` extra_func.

    Userargs:
      - bad_ics (list[int]): components to exclude (default ``[]``).
      - load_from_disk (bool): force a reload of the saved ICA even if one is
        already on ``dataset['ica']`` (default False).
    """
    userargs = proc_userargs(userargs, {
        'bad_ics': [],
        'load_from_disk': False,
    })

    missing = [k for k in ('subject', 'target_pth') if dataset.get(k) is None]
    if missing:
        raise KeyError(
            f"apply_ica needs dataset{missing} -- set them in your project "
            f"'initialize' extra_func (subject names the ICA folder, "
            f"target_pth is its root).")

    subject  = dataset['subject']
    ica_path = dataset['target_pth'] / subject / f'{subject}_ica.fif'

    if userargs['load_from_disk'] or 'ica' not in dataset:
        log_or_print(f'[apply_ica] Loading ICA from {ica_path}')
        ica = mne.preprocessing.read_ica(str(ica_path))
        dataset['ica'] = ica
    else:
        ica = dataset['ica']

    bad_ics = list(userargs['bad_ics'])
    ica.exclude = bad_ics
    log_or_print(f'[apply_ica] Excluding ICs {bad_ics} and applying to raw.')

    dataset['raw'] = ica.apply(dataset['raw'].copy())
    ica.save(str(ica_path), overwrite=True)
    return dataset
