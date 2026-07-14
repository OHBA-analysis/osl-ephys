import copy
from datetime import datetime
from functools import partial

import numpy as np
from scipy.stats import kurtosis
import mne
from osl_ephys.report.preproc_report import plot_channel_dists
from osl_ephys.utils.logger import log_or_print

from osl_ephys.preprocessing.semp.utils import ensure_dir, proc_userargs, require_keys
from ..metric import EEGTracer, psd_band_ratio, psd_band_stat
from ..vis import psd_plot, temp_plot, temp_plot_diff, pcs_plot


def init_tracer(dataset, userargs):
    """Initializes the EEGTracer with specific metrics for the dataset.

    Note: unlike the other wrappers this does **not** use
    ``proc_userargs(strict=True)`` -- ``userargs`` here is an open-ended set of
    ``metric_name -> callable`` pairs merged straight into the tracer, so there
    is no fixed key set to validate against.
    """
    tracer_kwargs = {
        "psd_mean": partial(psd_band_stat, band=[1, 40], fn=np.mean),
        "psd_kurtosis": partial(psd_band_stat, band=[1, 40], fn=kurtosis),
        "psd_maxmed_ratio": partial(psd_band_ratio, band1=[1, 40], fn1=np.max, band2=[1, 40], fn2=np.median),
        "psd_alpha_mean": partial(psd_band_stat, band='alpha', fn=np.mean),
        "psd_alpha_kurtosis": partial(psd_band_stat, band='alpha', fn=kurtosis),
        "psd_alpha_maxmed_ratio": partial(psd_band_ratio, band1='alpha', fn1=np.max, band2='alpha', fn2=np.median),
        "psd_beta_mean": partial(psd_band_stat, band='beta', fn=np.mean),
        "psd_beta_kurtosis": partial(psd_band_stat, band='beta', fn=kurtosis),
        "psd_beta_maxmed_ratio": partial(psd_band_ratio, band1='beta', fn1=np.max, band2='beta', fn2=np.median),
    }

    tracer_kwargs.update(dataset.get('tracer', {}))  # Update with any pre-set tracer metrics from dataset
    tracer_kwargs.update(userargs)

    dataset['tracer'] = EEGTracer(**tracer_kwargs)

    return dataset


def summary(dataset, userargs):
    """Generates a summary of the dataset, including basic statistics and channel information.
    Currently only plot the tracer checkpoints."""
    proc_userargs(userargs, {})   # takes no args; reject typos
    require_keys(dataset, ['subject', 'target_pth'], 'summary')

    subject = dataset['subject']
    if 'tracer' in dataset:
        dataset['tracer'].plot(save_pth=dataset['target_pth'] / "ckpt" / subject, show=False)

    # dataset.pop('tracer', None)  # Remove tracer from dataset after plotting
    return dataset


def _plot_psd_panel(dataset, userargs, save_fdr, fs):
    """The checkpoint PSD figure. If a per-key picks list exists
    (``picks_<key_to_print>``, written by epoch_aas / epoch_obs) the *returned*
    psd is over those channels and a separate ``psd.pdf`` over 'eeg' is also
    written; otherwise the eeg psd is both returned and saved. Returns
    ``(psd, picks)``."""
    if f"picks_{userargs['key_to_print']}" in dataset:
        picks = dataset[f"picks_{userargs['key_to_print']}"]
        psd = psd_plot(dataset['raw'], resolution=userargs['resolution'], fs=fs, figsize=userargs['psd_figsize'], fmax=userargs['max_freq'], picks=picks, dB=userargs['dB'])
        psd_plot(dataset['raw'], resolution=userargs['resolution'], fs=fs, figsize=userargs['psd_figsize'], fmax=userargs['max_freq'], save_pth=save_fdr / f"psd.pdf", picks='eeg', dB=userargs['dB'])
    else:
        picks = 'eeg'
        psd = psd_plot(dataset['raw'], resolution=userargs['resolution'], fs=fs, figsize=userargs['psd_figsize'], fmax=userargs['max_freq'], save_pth=save_fdr / f"psd.pdf", picks='eeg', dB=userargs['dB'])
    return psd, picks


def _plot_channel_dist(dataset, userargs, save_fdr):
    """Channel-std distribution panel (the mean std goes in the filename)."""
    std = np.mean(np.std(dataset['raw'].get_data(picks=userargs['std_channel_pick'], reject_by_annotation='omit'), axis=1))
    plot_channel_dists(dataset['raw'], str(save_fdr / f"std={std:.4e}.pdf"))


def _print_channel(dataset, ch_name, save_fdr, userargs, fs):
    """Per-channel PSD + time plots (full + focus range), plus a diff against
    the previous checkpoint's raw when one is stored.

    Swallows a narrow set of per-channel failures -- a channel name typo
    (KeyError), a bad slice (ValueError), an MNE plotting failure
    (RuntimeError), or a missing checkpoint dir (FileNotFoundError) -- so one
    bad channel does not abort the whole report. Anything else
    (KeyboardInterrupt, OOM, programmer errors) propagates.
    """
    try:
        extra_str = "channels"
        print_fdr = save_fdr / extra_str
        ensure_dir(print_fdr)

        psd_plot(dataset['raw'], resolution=userargs['resolution'], fs=fs, figsize=userargs['psd_figsize'], fmax=userargs['max_freq'], picks=ch_name, save_pth=print_fdr / f"{ch_name}_psd.pdf", dB=userargs['dB'])
        temp_plot(dataset['raw'], ch_name, fs=fs, save_pth=print_fdr / f"{ch_name}.pdf", name=ch_name)
        temp_plot(dataset['raw'], ch_name, fs=fs, start=userargs['focus_range'][0]*fs, length=(userargs['focus_range'][1]-userargs['focus_range'][0])*fs, save_pth=print_fdr / f"{ch_name}_{userargs['focus_range'][0]}-{userargs['focus_range'][1]}.pdf", name=ch_name)

        ### Compare with last checkpoint if exists
        if 'last_ckpt_raw' in dataset:
            if dataset['last_ckpt_raw'].info['sfreq'] != dataset['raw'].info['sfreq']:
                dataset['last_ckpt_raw'].resample(dataset['raw'].info['sfreq'])
            temp_plot_diff(dataset['last_ckpt_raw'], dataset['raw'], ch_name, fs=fs, save_pth=print_fdr / f"{ch_name}_diff.pdf", name=ch_name)
            temp_plot_diff(dataset['last_ckpt_raw'], dataset['raw'], ch_name, fs=fs, start=userargs['focus_range'][0]*fs, length=(userargs['focus_range'][1]-userargs['focus_range'][0])*fs, save_pth=print_fdr / f"{ch_name}_{userargs['focus_range'][0]}-{userargs['focus_range'][1]}_diff.pdf", name=ch_name)
    except (KeyError, ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"Error in printing channel {ch_name}: {type(e).__name__}: {e}")


def _select_channels(dataset, userargs, psd):
    """Channels to render: reuse the previous checkpoint's picks while they are
    still valid (so the same channels are tracked across the pipeline), else
    3 random channels (remembered for next time). ``always_print`` is unioned in.
    """
    channel_to_print = psd.ch_names
    if 'last_ckpt_print_ch' in dataset and set(dataset['last_ckpt_print_ch']).issubset(set(channel_to_print)):
        channel_to_print = dataset['last_ckpt_print_ch']
    elif len(channel_to_print) > 3:
        channel_to_print = np.random.choice(channel_to_print, 3, replace=False)
        dataset['last_ckpt_print_ch'] = channel_to_print

    return np.unique(np.concatenate([np.array(userargs['always_print']), channel_to_print]))


def _print_pcs(dataset, userargs, subject, channel_to_print, psd):
    """Render the AAS/OBS principal-component templates (``pc_<key_to_print>``)."""
    pc_fdr_name = dataset['target_pth'] / "ckpt" / subject / f"pc_{userargs['key_to_print']}"
    ensure_dir(pc_fdr_name)
    pcs_plot(dataset[f"pc_{userargs['key_to_print']}"], pc_fdr_name, channel_to_print, psd.ch_names, info=psd.info, resolution=userargs['resolution'], psd_lim=(0, userargs['max_freq']))


def _print_noise(dataset, userargs, subject, channel_to_print, fs):
    """Render the removed-noise raw (``noise_<key_to_print>``): per-channel PSD
    + time plots, plus a combined eeg PSD."""
    noise_fdr_name = dataset['target_pth'] / "ckpt" / subject / f"noise_{userargs['key_to_print']}"
    ensure_dir(noise_fdr_name)
    for ch_name in channel_to_print:
        psd_plot(dataset[f"noise_{userargs['key_to_print']}"], resolution=userargs['resolution'], fs=fs, figsize=userargs['psd_figsize'], fmax=userargs['max_freq'], picks=[ch_name], save_pth=noise_fdr_name / f"{ch_name}_psd.pdf", dB=userargs['dB'])
        temp_plot(dataset[f"noise_{userargs['key_to_print']}"], ch_name, fs=fs, save_pth=noise_fdr_name / f"{ch_name}.pdf", name=ch_name)

    psd_plot(dataset[f"noise_{userargs['key_to_print']}"], resolution=userargs['resolution'], fs=fs, figsize=userargs['psd_figsize'], fmax=userargs['max_freq'], save_pth=noise_fdr_name / f"noise_psd.pdf", picks='eeg', dB=userargs['dB'])


def _log_tracer(dataset, userargs, psd, picks):
    """Log this checkpoint's EEG + PSD into the EEGTracer, **if one exists**.

    The tracer is optional: it is only present when a project ran ``init_tracer``
    (or seeded a ``tracer`` dict in initialize). With no tracer, ckpt_report just
    skips the logging -- it does not require one.
    """
    if 'tracer' not in dataset:
        return
    if picks in ['eeg', 'all', 'data'] or 'eeg' in picks:
        dataset['tracer'].checkpoint(dataset['raw'].get_data(picks='eeg'), name=userargs['ckpt_name'])
        dataset['tracer'].checkpoint_psd(psd, name=userargs['ckpt_name'])
    else:
        log_or_print(f"Warning: EEG channels not fully included in picks for checkpointing tracer. Current picks: {picks}. Tracer logging skipped for this checkpoint.")


def ckpt_report(dataset, userargs):
    """a function for debugging the preprocessing steps.
        strictly requires Python >=3.7, for dict keys ordering

    Orchestrates the checkpoint panels; the actual rendering lives in the
    ``_plot_psd_panel`` / ``_plot_channel_dist`` / ``_print_channel`` /
    ``_print_pcs`` / ``_print_noise`` / ``_log_tracer`` helpers above.

    Args:
        dataset (dict): the dict containing raw data and metadata
        userargs (dict): a dictionary containing the optional arguments

    Returns:
        dataset: the updated dataset with the extra metadata
    """
    default_args = {
        'ckpt_name': datetime.now().strftime("%H:%M:%S"),
        'resolution': 0.05,
        'max_freq': 50,
        'key_to_print': None,
        # 'always_print': ['EKG'],    # must be name, 'eeg' is not allowed
        'always_print': [],    # must be name, 'eeg' is not allowed
        'std_channel_pick': 'eeg',
        'print_pcs': True,
        'print_noise': True,
        'dB': False,  # whether to plot psd in dB scale
        'focus_range': [100, 110],  # in seconds, for temp_plot
        'log_tracer': True,
        'psd_figsize': (10, 3),
    }
    userargs = proc_userargs(userargs, default_args)
    require_keys(dataset, ['subject', 'target_pth'], 'ckpt_report')

    fs = dataset['raw'].info['sfreq']
    subject = dataset['subject']
    save_fdr = dataset['target_pth'] / "ckpt" / subject / userargs['ckpt_name']
    ensure_dir(save_fdr)

    if userargs['key_to_print'] is None:
        userargs['print_noise'] = userargs['print_pcs'] = False

    psd, picks = _plot_psd_panel(dataset, userargs, save_fdr, fs)
    _plot_channel_dist(dataset, userargs, save_fdr)

    channel_to_print = _select_channels(dataset, userargs, psd)
    for ch_name in channel_to_print:
        _print_channel(dataset, ch_name, save_fdr, userargs, fs)

    ### Print PCs of OBS or AAS if requested.
    if userargs['print_pcs']:
        _print_pcs(dataset, userargs, subject, channel_to_print, psd)

    ### Print noise components if requested.
    if userargs['print_noise']:
        _print_noise(dataset, userargs, subject, channel_to_print, fs)

    ### Store the current raw data for diff comparison in the next checkpoint
    dataset['last_ckpt_raw'] = copy.deepcopy(dataset['raw'])

    ### Log tracer metrics if requested
    if userargs['log_tracer']:
        _log_tracer(dataset, userargs, psd, picks)

    return dataset
