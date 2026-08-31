# -*- coding: utf-8 -*-

"""
Preprocessing simultaneous EEG-fMRI with SEMP
=============================================

1. Set up files for the SEMP tutorial
-------------------------------------

This is an OSL-Ephys SEMP tutorial using NATVIEW. We use osl-pathfinder to keep the paired ``checker``, MR-artifact-free ``checkerout``, metadata, and output paths aligned by subject/session.
"""

#%%
# Install SEMP and download NATVIEW
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Start from a new environment, install the ``semp`` branch of `OSL-Ephys <https://github.com/OHBA-analysis/osl-ephys/tree/semp>`_.
#
# NATVIEW is openly available from ``s3://fcp-indi/data/Projects/NATVIEW_EEGFMRI/``. This tutorial needs only one or two complete subject/session directories. Install the AWS command-line client, choose a local data root, and download two example sessions without an AWS account:
#
# .. code-block:: bash
#
#     python -m pip install awscli
#     mkdir -p /path/to/natview/raw_data
#
#     aws s3 sync \
#       s3://fcp-indi/data/Projects/NATVIEW_EEGFMRI/raw_data/sub-01/ses-01/ \
#       /path/to/natview/raw_data/sub-01/ses-01/ \
#       --no-sign-request
#
#     aws s3 sync \
#       s3://fcp-indi/data/Projects/NATVIEW_EEGFMRI/raw_data/sub-05/ses-01/ \
#       /path/to/natview/raw_data/sub-05/ses-01/ \
#       --no-sign-request
#
# You may omit the second command for a one-recording walkthrough or replace the subject/session numbers with another available pair. Downloading the complete session keeps the EEG sidecars, events, channel table, BOLD metadata, and T1w image together. We read the EEGLAB ``.set`` file because the public release does not include the BrainVision ``.eeg`` binary. For the complete project contents and full-dataset download routes, see the `NATVIEW project page <https://fcon_1000.projects.nitrc.org/indi/retro/nat_view.html>`_.

#%%
# Configure the two project roots
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Edit these constants before running the notebook. Keep outputs separate from the downloaded NATVIEW tree.

#%%
NATVIEW_RAW_ROOT = '/path/to/natview/raw_data'
SEMP_TUTORIAL_RESULTS_ROOT = '/path/to/semp_tutorial_results'

print('NATVIEW_RAW_ROOT:', NATVIEW_RAW_ROOT)
print('SEMP_TUTORIAL_RESULTS_ROOT:', SEMP_TUTORIAL_RESULTS_ROOT)

#%%
# Describe the files used by SEMP
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``checker`` is the anchor that fixes the subject/session cohort. Each ``{foo}`` independently absorbs unimportant BIDS filename text and is never part of the recording ID. In a multi-file analysis project, move this definition into ``pathfinder.py`` and import its shared ``pf``.

#%%
from osl_pathfinder import Pathfinder

pf = Pathfinder(
    templates={
        'checker': (
            NATVIEW_RAW_ROOT
            + '/sub-{subject:02d}/ses-{session:02d}/eeg/'
            + '{foo}-checker_eeg.set'
        ),
        'checkerout': (
            NATVIEW_RAW_ROOT
            + '/sub-{subject:02d}/ses-{session:02d}/eeg/'
            + '{foo}-checkerout_eeg.set'
        ),
        'checker_eeg_json': (
            NATVIEW_RAW_ROOT
            + '/sub-{subject:02d}/ses-{session:02d}/eeg/'
            + '{foo}-checker_eeg.json'
        ),
        'checker_channels': (
            NATVIEW_RAW_ROOT
            + '/sub-{subject:02d}/ses-{session:02d}/eeg/'
            + '{foo}-checker_channels.tsv'
        ),
        'checker_events': (
            NATVIEW_RAW_ROOT
            + '/sub-{subject:02d}/ses-{session:02d}/eeg/'
            + '{foo}-checker_events.tsv'
        ),
        'checkerout_events': (
            NATVIEW_RAW_ROOT
            + '/sub-{subject:02d}/ses-{session:02d}/eeg/'
            + '{foo}-checkerout_events.tsv'
        ),
        'checker_bold_json': (
            NATVIEW_RAW_ROOT
            + '/sub-{subject:02d}/ses-{session:02d}/func/'
            + '{foo}-checker_bold.json'
        ),
        'checker_preproc': (
            SEMP_TUTORIAL_RESULTS_ROOT
            + '/checker/{subject}{session}/'
            + '{subject}{session}_preproc-raw.fif'
        ),
        'checkerout_preproc': (
            SEMP_TUTORIAL_RESULTS_ROOT
            + '/checkerout/{subject}{session}/'
            + '{subject}{session}_preproc-raw.fif'
        ),
    },
    id='{subject:d}{session:1d}',
    anchor='checker',
)

#%%
# Minimal Pathfinder API used here
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# - ``pf.ids`` --- fixed checker recording IDs discovered at construction.
# - ``pf.id2field(file_id)`` --- recover subject/session strings.
# - ``pf.id2path(file_id, kind)`` --- resolve an existing input path.
# - ``pf.id2path(file_id, kind, require_existence=False)`` --- construct a
#   future output path.
# - ``pf.path2id(path, kind)`` --- recover the recording ID from an input
#   filename.
# - ``pf.scan()`` --- validate all configured kinds and return a table.

#%%
ids = sorted(pf.ids)
if not ids:
    raise FileNotFoundError('No checker recordings match NATVIEW_RAW_ROOT')

file_id = ids[0]
print('checker recordings:', len(ids))
print('example ID and fields:', file_id, pf.id2field(file_id))
print('checker:', pf.id2path(file_id, 'checker'))
print('future output:', pf.id2path(file_id, 'checker_preproc', require_existence=False))

paired_ids = []
for candidate in ids:
    try:
        pf.id2path(candidate, 'checkerout')
    except FileNotFoundError:
        continue
    paired_ids.append(candidate)
print('paired checker/checkerout recordings:', len(paired_ids))

#%%
# 2. Explore one NATVIEW recording
# --------------------------------
#
# The purpose of this section is to fill the project ``initialize`` function.
# Four entries must be learned from the acquisition; the others come directly
# from the loaded file or the project objects already defined in Part 1.
#
# .. list-table:: Values stored by ``initialize``
#    :header-rows: 1
#
#    * - Dataset entry
#      - Source
#      - Used for
#    * - ``tr_interval``
#      - BOLD ``RepetitionTime``, checked against volume-trigger spacing
#      - TR epoch duration and scanner-volume crop
#    * - ``slice_interval``
#      - Spacing in BOLD ``SliceTiming``
#      - Locating residual slice-frequency peaks
#    * - ``tr_event_key``
#      - MNE annotation label for each acquired volume
#      - Selecting volume triggers
#    * - ``he_event_key``
#      - MNE annotation label for a helium-pump trigger, or ``[]`` if absent
#      - Optional helium-pump epoching
#    * - ``target_pth``
#      - Output root passed to ``initialize``
#      - Checkpoint and summary files
#    * - ``pf``
#      - Pathfinder constructed in Part 1
#      - File and recording-ID lookup
#    * - ``subject``
#      - ``pf.path2id`` applied to the input filename
#      - Per-recording output name
#    * - ``orig_sfreq``
#      - ``raw.info['sfreq']``
#      - Sampling-rate provenance
#    * - ``tracer``
#      - Optional functions for monitoring data cleaness
#      - e.g. variance, psd peak at slice-frequency, psd kurtosis, etc.

#%%
# 2.1 Select one recording
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Derive the acquisition constants from one recording first. Check additional
# recordings before applying the config to the full dataset.

#%%
import json
import numpy as np
import pandas as pd

import mne
from osl_ephys.preprocessing.semp import psd_plot, temp_plot

file_id = sorted(pf.ids)[0]
print('recording:', file_id, pf.id2field(file_id))

#%%
# 2.2 Load metadata and restore channel types
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# NATVIEW's channels TSV distinguishes EEG, EOG, and ECG channels. Apply those
# types after EEGLAB import so later ``picks='eeg'`` calls select EEG only.

#%%
checker_eeg = json.loads(pf.id2path(file_id, 'checker_eeg_json').read_text())
bold = json.loads(pf.id2path(file_id, 'checker_bold_json').read_text())
channels = pd.read_csv(pf.id2path(file_id, 'checker_channels'), sep='\t')
channel_type_map = dict(zip(channels['name'], channels['type'].str.lower()))
raw = mne.io.read_raw_eeglab(pf.id2path(file_id, 'checker'), preload=False, verbose='ERROR')
if set(raw.ch_names) != set(channel_type_map):
    raise ValueError('checker channel names do not match checker_channels.tsv')
raw.set_channel_types(channel_type_map)
print('raw sampling rate:', raw.info['sfreq'])
print('sidecar sampling rate:', checker_eeg['SamplingFrequency'])
print('MNE channel types:', pd.Series(raw.get_channel_types()).value_counts().to_dict())

#%%
# 2.3 Find ``tr_event_key`` and ``tr_interval``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# SEMP reads volume triggers from the loaded Raw annotations. For the selected
# recording, inspect the available labels, select the volume label, and verify
# that its median spacing agrees with the BOLD ``RepetitionTime``.

#%%
mne_events, mne_event_id = mne.events_from_annotations(raw, verbose='ERROR')
print('annotation labels:', sorted(mne_event_id))

TR_EVENT_KEY = ['R128']
if TR_EVENT_KEY[0] not in mne_event_id:
    raise ValueError('{} is absent from the Raw annotations'.format(TR_EVENT_KEY[0]))
tr_samples = np.unique(
    mne_events[mne_events[:, 2] == mne_event_id[TR_EVENT_KEY[0]], 0]
    - raw.first_samp
)
tr_onsets = tr_samples / raw.info['sfreq']
observed_tr_interval = float(np.median(np.diff(tr_onsets)))
TR_INTERVAL = float(bold['RepetitionTime'])
print('TR_EVENT_KEY:', TR_EVENT_KEY)
print('BOLD RepetitionTime: {:.4f} s'.format(TR_INTERVAL))
print('median trigger spacing: {:.4f} s'.format(observed_tr_interval))
if not np.isclose(TR_INTERVAL, observed_tr_interval, atol=0.5 / raw.info['sfreq']):
    raise ValueError('BOLD RepetitionTime and Raw trigger spacing disagree')

# The events TSV is not used as the trigger source. This selected-recording
# check only confirms that its relative R128 sequence agrees with the Raw
# annotations.
checker_events = pd.read_csv(pf.id2path(file_id, 'checker_events'), sep='\t')
tsv_r128 = checker_events.loc[
    checker_events['value'].astype(str).eq(TR_EVENT_KEY[0]), 'onset'
].dropna().to_numpy(float)
tsv_relative = tsv_r128 - tsv_r128[0]
mne_relative = tr_onsets - tr_onsets[0]
same_relative_timing = len(tsv_relative) == len(mne_relative) and np.allclose(
    tsv_relative, mne_relative, atol=0.5 / raw.info['sfreq']
)
print('events TSV matches relative Raw trigger timing:', same_relative_timing)

#%%
# 2.4 Find ``slice_interval``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``slice_reject`` uses this value to locate residual peaks near the slice
# frequency. Inspect the spacings between distinct entries in the BOLD
# ``SliceTiming`` array and use the most frequent spacing.

#%%
slice_timing = np.unique(np.round(np.asarray(bold['SliceTiming'], dtype=float), 6))
slice_steps = np.diff(slice_timing)
slice_steps = slice_steps[slice_steps > 0]
SLICE_INTERVAL = float(pd.Series(np.round(slice_steps, 4)).mode().iloc[0])
print('slice spacings:', pd.Series(np.round(slice_steps, 4)).value_counts().sort_index().to_dict())
print('SLICE_INTERVAL: {:.4f} s ({:.2f} Hz)'.format(SLICE_INTERVAL, 1 / SLICE_INTERVAL))

#%%
# 2.5 Find ``he_event_key``
# ^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A helium-pump event is optional. None of the selected recording's annotation
# labels represents that trigger, so this pipeline disables helium epoching.

#%%
HE_EVENT_KEY = []
print('HE_EVENT_KEY:', HE_EVENT_KEY)

#%%
# 2.6 Fill ``initialize``
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The four acquisition constants now come from the checks above. All remaining
# entries are derived directly from the current input and project objects.

#%%
from functools import partial
from osl_ephys.preprocessing.semp.metric import psd_band_ratio

def initialize(dataset, userargs):
    dataset['tr_interval'] = userargs.get('tr_interval', TR_INTERVAL)
    dataset['slice_interval'] = userargs.get('slice_interval', SLICE_INTERVAL)
    dataset['tr_event_key'] = userargs.get('tr_event_key', TR_EVENT_KEY)
    dataset['he_event_key'] = userargs.get('he_event_key', HE_EVENT_KEY)
    dataset['target_pth'] = userargs['target_pth']
    # Use the shared Pathfinder imported above; do not pass it through
    # the batch config/userargs, which are deep-copied by run_proc_batch.
    dataset['pf'] = pf
    dataset['subject'] = dataset['pf'].path2id(dataset['raw'].filenames[0], 'checker')
    dataset['orig_sfreq'] = dataset['raw'].info['sfreq']
    si = dataset['slice_interval']
    dataset['tracer'] = {
        'psd_slice': partial(psd_band_ratio, band1=[1 / si - 1, 1 / si + 1], band2='beta', fn1=np.mean),
        'psd_2slice': partial(psd_band_ratio, band1=[2 / si - 1, 2 / si + 1], band2=[20, 35], fn1=np.mean),
    }
    return dataset

print('initialize values:', TR_INTERVAL, SLICE_INTERVAL, TR_EVENT_KEY, HE_EVENT_KEY)

#%%
# 2.7 Plot the initial and reference spectra
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# These spectra are used by the sensor-space comparison after preprocessing.
# ``checkerout`` is the paired recording acquired outside the scanner; it is a
# subject-level reference, not a time-aligned target.

#%%
rawout = mne.io.read_raw_eeglab(
    pf.id2path(file_id, 'checkerout'), preload=False, verbose='ERROR'
)
if set(rawout.ch_names) != set(channel_type_map):
    raise ValueError('checkerout channel names do not match checker_channels.tsv')
rawout.set_channel_types(channel_type_map)

psd_checker = psd_plot(
    raw, name='checker | PSD', picks='eeg',
    fmin=1, fmax=125, resolution=0.05, dB=True,
)
temp_plot(raw, channel=raw.ch_names[0], name='checker | ' + raw.ch_names[0])

#%%
psd_checkerout = psd_plot(
    rawout, name='checkerout | PSD', picks='eeg',
    fmin=1, fmax=125, resolution=0.05, dB=True,
)
temp_plot(rawout, channel=rawout.ch_names[0], name='checkerout | ' + rawout.ch_names[0])

#%%
# 3. SEMP stages: remove one artefact at a time
# ---------------------------------------------
#
# Run the SEMP pipeline one stage at a time before using its batch config. The
# order is: define TR epochs, retain the scanner-on interval, subtract the
# gradient template, filter and resample, detect bad data, apply one ICA fit,
# then interpolate and re-reference.
#
# ``find_func`` resolves both SEMP stages and ordinary MNE/OSL stages, so every
# call below uses the same stage name and arguments as the final config.

#%%
# 3.1 Create one direct-stage dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Create the OSL-Ephys dataset and a helper that runs one config-style stage.

#%%
from pathlib import Path

from osl_ephys.preprocessing.batch import find_func

RESULTS_ROOT = SEMP_TUTORIAL_RESULTS_ROOT
PLOT_CHANNEL = 'Fp1'

dataset = {
    'raw': raw, 'events': None, 'epochs': None, 'event_id': None,
    'ica': None, 'fig': {},
}

def stage(name, **kwargs):
    """Run one stage directly, without a config or batch runner."""
    func = find_func(name, extra_funcs=[initialize])
    if func is None:
        raise RuntimeError('Could not resolve stage: {}'.format(name))
    print('-- {} {}'.format(name, kwargs))
    dataset.update(func(dataset, kwargs))
    return dataset

#%%
# Checkpoint plots
# ^^^^^^^^^^^^^^^^
#
# The initial, post-AAS, post-ICA, and final plots are sufficient for this
# walkthrough. Every PSD uses 0.05 Hz bins.

#%%
# 3.2 Stage 0: initialise the dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Add the project and acquisition values derived in Part 2.

#%%
stage('initialize', target_pth=Path(RESULTS_ROOT) / 'checker', tr_interval=TR_INTERVAL, slice_interval=SLICE_INTERVAL, tr_event_key=TR_EVENT_KEY)
print(dataset['subject'], dataset['tr_interval'], dataset['slice_interval'])

#%%
# 3.3 Name channels and preserve the embedded montage
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Restore ECG/EOG channel types and retain the sensor positions embedded in the
# EEGLAB file. The positions are required for ICA topographies.

#%%
stage('set_channel_types', ECG='ecg', EOGL='eog', EOGU='eog')
montage = dataset['raw'].get_montage()
if montage is None or not montage.get_positions()['ch_pos']:
    raise ValueError('NATVIEW input has no embedded sensor montage.')
print('embedded montage channels:', len(montage.get_positions()['ch_pos']))

#%%
# 3.4 Create TR epochs, crop scanner volumes, and remove line noise
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Use the fixed order ``create_TR_epoch -> crop_TR -> notch_filter``. Epochs are
# created from the complete trigger sequence before the recording boundaries
# change. ``preserve_epochs=True`` then retains the valid pre-created epochs
# through the crop. Inspect the epoch and trigger counts before continuing.

#%%
stage('create_TR_epoch')
expected_tr_samples = round(dataset['tr_interval'] * dataset['raw'].info['sfreq'])
assert len(dataset['tr_ep'].times) == expected_tr_samples
print('TR epochs:', len(dataset['tr_ep']), dataset['tr_ep'].get_data().shape)
print('first TR epoch relative onset (s):', (
    dataset['tr_ep'].events[0, 0] - dataset['raw'].first_samp
) / dataset['raw'].info['sfreq'])
print('last TR epoch relative onset (s):', (
    dataset['tr_ep'].events[-1, 0] - dataset['raw'].first_samp
) / dataset['raw'].info['sfreq'])

#%%
# 3.5 Crop, then notch
# ^^^^^^^^^^^^^^^^^^^^
#
# Retain the scanner-on interval, confirm how many R128 annotations remain, then
# remove the 60 and 120 Hz line components.

#%%
stage('crop_TR', preserve_epochs=True)
print('cropped duration:', dataset['raw'].times[-1])
cropped_events, cropped_event_id = mne.events_from_annotations(dataset['raw'], verbose='ERROR')
if 'R128' in cropped_event_id:
    cropped_events = cropped_events[cropped_events[:, 2] == cropped_event_id['R128']]
else:
    cropped_events = np.empty((0, 3), dtype=int)
print('cropped R128 count:', len(cropped_events))

stage('notch_filter', freqs='60 120')
psd_plot(
    dataset['raw'], name='after TR crop and notch filter | PSD', picks='eeg',
    fmin=0, fmax=125, resolution=0.05, dB=False,
)

#%%
# 3.6 Remove the gradient artefact with AAS
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# ``epoch_aas`` forms a sliding template from 30 TR epochs and subtracts it with
# ``fit=False``. The removed signal is retained as ``noise_tr_ep`` for QA.

#%%
stage('epoch_aas', epoch_key='tr_ep', overwrite='new', picks='all', window_length=30, fit=False)
print('AAS removed:', 'noise_tr_ep' in dataset)
psd_plot(
    dataset['raw'], name='after AAS | PSD', picks='eeg',
    fmin=0, fmax=125, resolution=0.05, dB=False,
)
temp_plot(dataset['raw'], channel=PLOT_CHANNEL, name='after AAS | ' + PLOT_CHANNEL)

#%%
# 3.7 Filter, edge-crop, and resample
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Filter only after AAS. Apply the 0.5-125 Hz IIR filter, remove five seconds
# from each filtered edge, then resample to 250 Hz.

#%%
stage('filter', l_freq=0.5, h_freq=125, method='iir', iir_params={'order': 5, 'ftype': 'butter'})
stage('mid_crop', edge=5)
stage('resample', sfreq=250)
print('new sfreq:', dataset['raw'].info['sfreq'])

#%%
# 3.8 Detect bad segments and channels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Mark noisy/flat intervals and channels. Bad-channel interpolation is deferred
# until after ICA.

#%%
stage('bad_segments', segment_len=500, picks='eeg', significance_level=0.1, detect_zeros=False)
stage('bad_segments', segment_len=500, picks='eeg', mode='diff', significance_level=0.1, detect_zeros=False)
stage('bad_channels', picks='eeg', significance_level=0.1)
stage('bad_segments', segment_len=2500, picks='eog', detect_zeros=False)
print('bad channels:', dataset['raw'].info['bads'])

#%%
# 3.9 One ICA fit, one combined apply
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Fit ICA once. ``ica_autoreject(apply=False)`` selects EOG/ECG components
# without applying them. ``slice_reject`` adds components selected by residual
# slice-frequency power, then applies the combined exclusion list once.

#%%
stage('ica_raw', n_components=0.999, picks='eeg', l_freq=1, random_state=42)
stage('ica_autoreject', eogmeasure='correlation', eogthreshold=0.35, ecgmethod='ctps', ecgthreshold=0.1, apply=False)
print('ICA exclusions before slice test:', dataset['ica'].exclude)
stage('slice_reject')   # slice_reject defaults to apply=True
print('ICA exclusions after slice test:', dataset['ica'].exclude)
psd_plot(
    dataset['raw'], name='after ICA and slice rejection | PSD', picks='eeg',
    fmin=0, fmax=125, resolution=0.05, dB=False,
)
temp_plot(
    dataset['raw'], channel=PLOT_CHANNEL,
    name='after ICA and slice rejection | ' + PLOT_CHANNEL,
)

#%%
# 3.10 Interpolate and re-reference
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Only after ICA has removed the selected components do we interpolate bad EEG channels, then apply the average reference directly.

#%%
stage('interpolate_bads')
stage('set_eeg_reference', projection=False)
print('final bad channels:', dataset['raw'].info['bads'])
psd_final = psd_plot(
    dataset['raw'], name='final re-referenced output | PSD', picks='eeg',
    fmin=0, fmax=125, resolution=0.05, dB=False,
)
temp_plot(
    dataset['raw'], channel=PLOT_CHANNEL,
    name='final re-referenced output | ' + PLOT_CHANNEL,
)

#%%
# 3.11 A compact sensor-space validation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# A successful run should suppress scanner harmonics without destroying the
# overall EEG spectrum. Compare the initial ``checker``, cleaned ``checker``,
# and paired ``checkerout`` spectra on the same 0.05 Hz grid. The correlation
# and R-squared below compare standardized log-PSD *shape*; they are not
# sample-wise accuracy measures because the paired recordings are not
# time-aligned.

#%%
import matplotlib.pyplot as plt

def mean_log_psd(spectrum, grid):
    """Interpolate the channel-mean log PSD onto one frequency grid."""
    power = np.mean(spectrum.get_data(), axis=0)
    power_db = 10 * np.log10(np.maximum(power, np.finfo(float).tiny))
    return np.interp(grid, spectrum.freqs, power_db)

grid = np.arange(1.0, 125.0 + 0.025, 0.05)
curves = {
    'checker before SEMP': mean_log_psd(psd_checker, grid),
    'checker after SEMP': mean_log_psd(psd_final, grid),
    'checkerout reference': mean_log_psd(psd_checkerout, grid),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, upper in zip(axes, (45, 125)):
    keep = grid <= upper
    for label, curve in curves.items():
        ax.plot(grid[keep], curve[keep], label=label)
    ax.set(xlabel='Frequency (Hz)', ylabel='Mean PSD (dB)', xlim=(1, upper))
    ax.grid(alpha=0.25)
axes[0].legend()
fig.tight_layout()

keep = grid <= 45
clean_shape = curves['checker after SEMP'][keep]
reference_shape = curves['checkerout reference'][keep]
clean_z = (clean_shape - clean_shape.mean()) / clean_shape.std()
reference_z = (reference_shape - reference_shape.mean()) / reference_shape.std()
spectral_r = float(np.corrcoef(clean_z, reference_z)[0, 1])
spectral_r2 = float(1 - np.sum((clean_z - reference_z) ** 2) / np.sum(reference_z ** 2))

slice_ratio_before = float(np.median(psd_band_ratio(
    psd_checker, band1=[1 / SLICE_INTERVAL - 1, 1 / SLICE_INTERVAL + 1],
    band2='beta', fn1=np.mean,
)))
slice_ratio_after = float(np.median(psd_band_ratio(
    psd_final, band1=[1 / SLICE_INTERVAL - 1, 1 / SLICE_INTERVAL + 1],
    band2='beta', fn1=np.mean,
)))
print('cleaned vs checkerout log-PSD shape: r={:.3f}, R^2={:.3f}'.format(
    spectral_r, spectral_r2,
))
print('median slice-band / beta ratio: {:.3f} -> {:.3f}'.format(
    slice_ratio_before, slice_ratio_after,
))

#%%
# 3.12 The batch version
# ^^^^^^^^^^^^^^^^^^^^^^
#
# The config below repeats the direct-stage order for every ``checker`` ID.
# Checkpoints and tracer values provide batch QA. Process ``checkerout`` with a
# separate ordinary-EEG reference config. For human-reviewed ICA, use the
# manual-ICA tutorial and defer interpolation/re-referencing until review has
# been applied.

#%%
from osl_ephys.preprocessing import run_proc_batch

# This is the same order used above. If you have downloaded more subjects, you can run this for a batch process
target_pth = Path(RESULTS_ROOT) / 'checker'
config = {
    'preproc': [
        {'initialize': {'target_pth': target_pth}},
        {'init_tracer': {}},
        {'set_channel_types': {'ECG': 'ecg', 'EOGL': 'eog', 'EOGU': 'eog'}},
        {'create_TR_epoch': {}},
        {'crop_TR': {'preserve_epochs': True}},
        {'ckpt_report': {'ckpt_name': 'raw', 'dB': False}},
        {'notch_filter': {'freqs': '60 120'}},
        {'epoch_aas': {
            'epoch_key': 'tr_ep', 'overwrite': 'new', 'picks': 'all',
            'window_length': 30, 'fit': False,
        }},
        {'ckpt_report': {
            'ckpt_name': 'after_aas_removal', 'key_to_print': 'tr_ep',
            'dB': False,
        }},
        {'filter': {
            'l_freq': 0.5, 'h_freq': 125, 'method': 'iir',
            'iir_params': {'order': 5, 'ftype': 'butter'},
        }},
        {'mid_crop': {'edge': 5}},
        {'resample': {'sfreq': 250}},
        {'ckpt_report': {'ckpt_name': 'after_filt', 'dB': False}},
        {'bad_segments': {
            'segment_len': 500, 'picks': 'eeg', 'significance_level': 0.1,
            'detect_zeros': False,
        }},
        {'bad_segments': {
            'segment_len': 500, 'picks': 'eeg', 'mode': 'diff',
            'significance_level': 0.1, 'detect_zeros': False,
        }},
        {'bad_channels': {'picks': 'eeg', 'significance_level': 0.1}},
        {'bad_segments': {
            'segment_len': 2500, 'picks': 'eog', 'detect_zeros': False,
        }},
        {'ica_raw': {
            'n_components': 0.999, 'picks': 'eeg', 'l_freq': 1,
            'random_state': 42,
        }},
        {'ica_autoreject': {
            'eogmeasure': 'correlation', 'eogthreshold': 0.35,
            'ecgmethod': 'ctps', 'ecgthreshold': 0.1, 'apply': False,
        }},
        {'slice_reject': {}},
        {'ckpt_report': {'ckpt_name': 'after_ica', 'dB': False}},
        {'bad_channels': {'picks': 'eeg', 'significance_level': 0.1}},
        {'interpolate_bads': {}},
        {'ckpt_report': {'ckpt_name': 'after_interp', 'dB': False}},
        {'set_eeg_reference': {'projection': False}},
        {'cleanup': {'keywords': ['noise_', 'pf']}},
        {'summary': {}},
    ]
}

subject_list = sorted(pf.ids)
file_list = [str(pf.id2path(file_id, 'checker')) for file_id in subject_list]
RUN_BATCH = False
if RUN_BATCH:
    run_proc_batch(
        config, file_list, subjects=subject_list, outdir=str(target_pth),
        extra_funcs=[initialize], gen_report=True, overwrite=False,
        random_seed=42,
    )
