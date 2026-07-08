"""Use MNE for source localisation.

https://mne.tools/stable/auto_tutorials/inverse/30_mne_dspm_loreta.html#sphx-glr-auto-tutorials-inverse-30-mne-dspm-loreta-py
https://mne.tools/stable/auto_examples/inverse/morph_surface_stc.html
"""

import os
import mne
import numpy as np
from mne import minimum_norm

def calc_noise_cov(raw, data_cov_rank=60, chantypes=None):
    """Calculate noise covariance."""
    # In MNE, the noise cov is normally obtained from empty room noise
    # recordings or from a baseline period. Here (if no noise cov is passed in)
    # we mimic what the osl_normalise_sensor_data.m function in Matlab OSL does,
    # by computing a diagonal noise cov with the variances set to the mean
    # variance of each sensor type (e.g. mag, grad, eeg.)
    if chantypes is None:
        chantypes = ["mag", "grad"]
    raw = raw.pick(chantypes)
    data_cov = mne.compute_raw_covariance(
        raw, method="empirical", rank=data_cov_rank
    )
    n_channels = data_cov.data.shape[0]
    noise_cov_diag = np.zeros(n_channels)
    for type in chantypes:
        # Indices of this channel type
        type_raw = raw.copy().pick(type, exclude="bads")
        inds = []
        for chan in type_raw.info["ch_names"]:
            inds.append(data_cov.ch_names.index(chan))

        # Mean variance of channels of this type
        variance = np.mean(np.diag(data_cov.data)[inds])
        noise_cov_diag[inds] = variance
        print(f"variance for chantype {type} is {variance}")

    bads = [b for b in raw.info["bads"] if b in data_cov.ch_names]
    noise_cov = mne.Covariance(
        noise_cov_diag, data_cov.ch_names, bads, raw.info["projs"], nfree=1e10
    )
    return noise_cov

def standardize(data):
    data -= np.mean(data, axis=-1, keepdims=True)
    data /= np.std(data, axis=-1, keepdims=True)
    return data

# Directories and files
subjects_dir = "../data"
subject = "sub-01"

stc_dir = f"{subjects_dir}/stc"

fwd_fname = f"{subjects_dir}/{subject}/model-fwd.fif"
src_fname = f"{subjects_dir}/{subject}/space-src.fif"

os.makedirs(stc_dir, exist_ok=True)

for run in range(1, 7):
    preproc_raw_fname = f"{subjects_dir}/preproc/{subject}_run-{run:02d}_preproc-raw.fif"
    #preproc_epochs_fname = f"{subjects_dir}/preproc-data/{subject}-preproc-epo.fif"
    #fsaverage_src_fname = f"{subjects_dir}/morph-maps/fsaverage-src.fif"

    # Preprocessed data
    raw = mne.io.read_raw_fif(preproc_raw_fname, preload=True)
    #epochs = mne.read_epochs(preproc_epochs_fname, preload=True)
    #evoked = epochs[["famous", "unfamiliar", "scrambled"]].average()
    #del epochs

    # Compute covariances
    noise_cov = calc_noise_cov(raw)
    #emptyroom_raw = mne.io.read_raw_fif("../data/raw/sub-emptyroom_ses-20090409_task-noise_proc-sss_meg.fif", preload=True)
    #noise_cov = mne.compute_raw_covariance(emptyroom_raw, method="empirical", rank=60)

    # Calculate inverse operator
    fwd = mne.read_forward_solution(fwd_fname)
    inverse_operator = minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
    del fwd

    # Source localise continuous data
    stc_raw = minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2=0.1, method="eLORETA")
    stc_raw.save(f"{stc_dir}/{subject}_raw", overwrite=True)

    # Source localise evoked data
    #stc_evoked = minimum_norm.apply_inverse(evoked, inverse_operator, lambda2=0.1, method="eLORETA")

    # Morph to a standard brain
    #src_orig = mne.read_source_spaces(src_fname)
    #src_to = mne.read_source_spaces(fsaverage_src_fname)

    #morph = mne.compute_source_morph(
    #    stc_raw,
    #    subject_from=subject,
    #    subject_to="fsaverage",
    #    src_to=src_to,
    #    subjects_dir=subjects_dir,
    #)

    #stc_raw_fsaverage = morph.apply(stc_raw)
    #stc_raw_fsaverage.save(f"{stc_dir}/{subject}-raw", overwrite=True)

    #stc_evoked_fsaverage = morph.apply(stc_evoked)
    #stc_evoked_fsaverage.apply_function(standardize)
    #stc_evoked_fsaverage.apply_baseline(baseline=(None, 0))
    #stc_evoked_fsaverage.save(f"{stc_dir}/{subject}-epo", overwrite=True)
