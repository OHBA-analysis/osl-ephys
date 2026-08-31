# -*- coding: utf-8 -*-

"""
Manual ICA review for the NATVIEW SEMP pipeline
===============================================

This tutorial uses the same NATVIEW ``sub-01/ses-01`` recording, Pathfinder,
``initialize`` function, and pre-ICA stages as the simultaneous EEG-fMRI
tutorial. The difference begins at ICA: ``manual_ica`` fits and renders the
components, but does not apply any exclusions. A user reviews the components
before a separate command creates the cleaned file.
"""

#%%
# 1. Download the same NATVIEW recording
# --------------------------------------
#
# Skip this download if ``sub-01/ses-01`` was already downloaded for the
# preceding EEG-fMRI tutorial.
#
# .. code-block:: bash
#
#     python -m pip install awscli
#     mkdir -p /path/to/natview/raw_data
#     aws s3 sync \
#       s3://fcp-indi/data/Projects/NATVIEW_EEGFMRI/raw_data/sub-01/ses-01/ \
#       /path/to/natview/raw_data/sub-01/ses-01/ \
#       --no-sign-request

#%%
# Configure the same project roots
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#%%
NATVIEW_RAW_ROOT = '/path/to/natview/raw_data'
SEMP_TUTORIAL_RESULTS_ROOT = '/path/to/semp_tutorial_results'

print('NATVIEW_RAW_ROOT:', NATVIEW_RAW_ROOT)
print('SEMP_TUTORIAL_RESULTS_ROOT:', SEMP_TUTORIAL_RESULTS_ROOT)

#%%
# Use the same Pathfinder
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The input and metadata templates are identical to the automatic tutorial.
# Three additional kinds describe the manual-ICA output files.

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
        'checker_manual_preproc': (
            SEMP_TUTORIAL_RESULTS_ROOT
            + '/checker_manual/{subject}{session}/'
            + '{subject}{session}_preproc-raw.fif'
        ),
        'checker_manual_ica': (
            SEMP_TUTORIAL_RESULTS_ROOT
            + '/checker_manual/{subject}{session}/'
            + '{subject}{session}_ica.fif'
        ),
        'checker_manual_after_ica': (
            SEMP_TUTORIAL_RESULTS_ROOT
            + '/checker_manual/{subject}{session}/'
            + '{subject}{session}_after_ica-raw.fif'
        ),
    },
    id='{subject:d}{session:1d}',
    anchor='checker',
)

file_id = '11'  # sub-01/ses-01
if file_id not in pf.ids:
    raise FileNotFoundError('sub-01/ses-01 checker data were not found')
print('checker:', pf.id2path(file_id, 'checker'))
print('future manual output:', pf.id2path(
    file_id, 'checker_manual_preproc', require_existence=False,
))

#%%
# 2. Run the same pipeline up to manual ICA
# -----------------------------------------
#
# These acquisition values are the values derived for NATVIEW in the EEG-fMRI
# tutorial. The initializer is the same: it attaches acquisition timing,
# project paths, the recording ID, and optional QA metrics.

#%%
from functools import partial
from pathlib import Path

import numpy as np

from osl_ephys.preprocessing import run_proc_batch
from osl_ephys.preprocessing.semp.metric import psd_band_ratio

TR_INTERVAL = 2.1
SLICE_INTERVAL = 0.055
TR_EVENT_KEY = ['R128']
HE_EVENT_KEY = []


def initialize(dataset, userargs):
    dataset['tr_interval'] = userargs.get('tr_interval', TR_INTERVAL)
    dataset['slice_interval'] = userargs.get(
        'slice_interval', SLICE_INTERVAL
    )
    dataset['tr_event_key'] = userargs.get('tr_event_key', TR_EVENT_KEY)
    dataset['he_event_key'] = userargs.get('he_event_key', HE_EVENT_KEY)
    dataset['target_pth'] = userargs['target_pth']
    dataset['pf'] = pf
    dataset['subject'] = pf.path2id(
        dataset['raw'].filenames[0], 'checker'
    )
    dataset['orig_sfreq'] = dataset['raw'].info['sfreq']
    si = dataset['slice_interval']
    dataset['tracer'] = {
        'psd_slice': partial(
            psd_band_ratio,
            band1=[1 / si - 1, 1 / si + 1],
            band2='beta',
            fn1=np.mean,
        ),
        'psd_2slice': partial(
            psd_band_ratio,
            band1=[2 / si - 1, 2 / si + 1],
            band2=[20, 35],
            fn1=np.mean,
        ),
    }
    return dataset


target_pth = Path(SEMP_TUTORIAL_RESULTS_ROOT) / 'checker_manual'
ica_review_pth = Path(SEMP_TUTORIAL_RESULTS_ROOT) / 'checker_manual_ica_review'

config = {
    'preproc': [
        {'initialize': {'target_pth': target_pth}},
        {'init_tracer': {}},
        {'set_channel_types': {
            'ECG': 'ecg', 'EOGL': 'eog', 'EOGU': 'eog',
        }},
        {'create_TR_epoch': {}},
        {'crop_TR': {'preserve_epochs': True}},
        {'ckpt_report': {'ckpt_name': 'raw', 'dB': False}},
        {'notch_filter': {'freqs': '60 120'}},
        {'epoch_aas': {
            'epoch_key': 'tr_ep',
            'overwrite': 'new',
            'picks': 'all',
            'window_length': 30,
            'fit': False,
        }},
        {'ckpt_report': {
            'ckpt_name': 'after_aas_removal',
            'key_to_print': 'tr_ep',
            'dB': False,
        }},
        {'filter': {
            'l_freq': 0.5,
            'h_freq': 125,
            'method': 'iir',
            'iir_params': {'order': 5, 'ftype': 'butter'},
        }},
        {'mid_crop': {'edge': 5}},
        {'resample': {'sfreq': 250}},
        {'ckpt_report': {'ckpt_name': 'after_filt', 'dB': False}},
        {'bad_segments': {
            'segment_len': 500,
            'picks': 'eeg',
            'significance_level': 0.1,
            'detect_zeros': False,
        }},
        {'bad_segments': {
            'segment_len': 500,
            'picks': 'eeg',
            'mode': 'diff',
            'significance_level': 0.1,
            'detect_zeros': False,
        }},
        {'bad_channels': {
            'picks': 'eeg',
            'significance_level': 0.1,
        }},
        {'bad_segments': {
            'segment_len': 2500,
            'picks': 'eog',
            'detect_zeros': False,
        }},
        {'manual_ica': {
            'outdir': ica_review_pth,
            'n_components': 0.999,
            'picks': 'eeg',
            'l_freq': 1.0,
            'seed': 42,
            'psd_resolution': 0.05,
        }},
        # This removes non-signal QA objects before run_proc_batch serializes
        # the dataset. It does not alter the fitted ICA or preprocessed Raw.
        {'cleanup': {'keywords': ['noise_', 'pf']}},
    ]
}

subject_list = [file_id]
file_list = [str(pf.id2path(file_id, 'checker'))]

RUN_BATCH = False
if RUN_BATCH:
    run_proc_batch(
        config,
        file_list,
        subjects=subject_list,
        outdir=str(target_pth),
        extra_funcs=[initialize],
        gen_report=False,
        overwrite=False,
        random_seed=42,
    )

#%%
# The fit batch intentionally stops here. It writes:
#
# - ``checker_manual/11/11_preproc-raw.fif``;
# - ``checker_manual/11/11_ica.fif``;
# - ``checker_manual_ica_review/11/`` containing the review pages and labels.
#
# It does not run ``ica_autoreject``, ``slice_reject``, a second bad-channel
# pass, interpolation, or re-referencing. ``manual_ica`` displays EOG, ECG, and
# residual gradient-artefact scores as review aids, but applies no components.

#%%
# 3. Review the fitted components
# --------------------------------
#
# Serve the review root:
#
# .. code-block:: bash
#
#     cd /path/to/semp_tutorial_results/checker_manual_ica_review
#     osl-ica-review 8000
#
# Open ``http://localhost:8000/11/single_ic.html`` and label every component
# ``good``, ``bad``, or ``unsure``. Bad time intervals can also be recorded.

#%%
# 4. Apply the reviewed decisions
# --------------------------------
#
# Apply subject 11 after every component has a label:
#
# .. code-block:: bash
#
#     osl-ica-apply \
#       /path/to/semp_tutorial_results/checker_manual_ica_review \
#       /path/to/semp_tutorial_results/checker_manual \
#       11
#
# This reads ``11_preproc-raw.fif`` and ``11_ica.fif``, applies components
# labelled bad, adds reviewed bad intervals as annotations, and writes
# ``11_after_ica-raw.fif``. Incomplete or malformed reviews are skipped.

#%%
# 5. Interpolate and re-reference after review
# --------------------------------------------
#
# These operations are intentionally outside the fit batch because they belong
# after the reviewed ICA exclusions have been applied.

#%%
import mne

RUN_FINISH = False
if RUN_FINISH:
    cleaned_path = pf.id2path(file_id, 'checker_manual_after_ica')
    cleaned = mne.io.read_raw_fif(cleaned_path, preload=True)
    cleaned.interpolate_bads()
    cleaned.set_eeg_reference('average', projection=False)
    cleaned.save(cleaned_path, overwrite=True)

#%%
# The public Python API provides the same apply operation when a project needs
# custom orchestration:

#%%
from osl_ephys.preprocessing.manual_ica import apply_manual_ica

RUN_PYTHON_APPLY = False
if RUN_PYTHON_APPLY:
    status = apply_manual_ica(
        ica_review_pth,
        target_pth,
        file_id,
        overwrite=False,
        purge_svgs=False,
    )
    print(status)
