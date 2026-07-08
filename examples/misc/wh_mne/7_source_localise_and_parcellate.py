"""Use MNE for source localisation and parcellation.

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

def save_parc_data(data, raw, output_file, ch_names=None, extra_chans="stim"):
    """Save parcellated data as a fif file."""
    if isinstance(raw, str):
        raw = mne.io.read_raw_fif(raw)

    # What extra channels should we add to the new raw object?
    if isinstance(extra_chans, str):
        extra_chans = [extra_chans]
    extra_chans = np.unique(["stim"] + extra_chans)

    # Create Info object
    info = raw.info
    if ch_names is None:
        ch_names = [f"ch_{i}" for i in range(data.shape[0])]
    parc_info = mne.create_info(ch_names=ch_names, ch_types="misc", sfreq=info["sfreq"])

    # Create Raw object
    new_raw = mne.io.RawArray(data, parc_info)

    # Copy timing info
    new_raw.set_meas_date(raw.info["meas_date"])
    new_raw.__dict__["_first_samps"] = raw.__dict__["_first_samps"]
    new_raw.__dict__["_last_samps"] = raw.__dict__["_last_samps"]
    new_raw.__dict__["_cropped_samp"] = raw.__dict__["_cropped_samp"]

    # Copy annotations from raw
    new_raw.set_annotations(raw._annotations)

    # Add extra channels
    if "stim" not in raw:
        log_or_print("No stim channel to add to fif file", warning=True)
    for extra_chan in extra_chans:
        if extra_chan in raw:
            chan_raw = raw.copy().pick(extra_chan)
            chan_data = chan_raw.get_data()
            chan_info = mne.create_info(chan_raw.ch_names, raw.info["sfreq"], [extra_chan] * chan_data.shape[0])
            chan_raw = mne.io.RawArray(chan_data, chan_info)
            new_raw.add_channels([chan_raw], force_update_info=True)

    # Copy the description from the sensor-level Raw object
    new_raw.info["description"] = raw.info["description"]

    # Save
    new_raw.save(output_file, overwrite=True)

# Directories and files
subjects_dir = "../data"
subject = "sub-01"

parc_dir = f"{subjects_dir}/parc_data"

fwd_fname = f"{subjects_dir}/{subject}/model-fwd.fif"
src_fname = f"{subjects_dir}/{subject}/space-src.fif"

os.makedirs(parc_dir, exist_ok=True)

for run in range(1, 7):
    preproc_raw_fname = f"{subjects_dir}/preproc/{subject}_run-{run:02d}_preproc-raw.fif"

    # Source localise continuous data
    raw = mne.io.read_raw_fif(preproc_raw_fname, preload=True)
    noise_cov = calc_noise_cov(raw)
    fwd = mne.read_forward_solution(fwd_fname)
    inverse_operator = minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
    del fwd
    stc = minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2=0.1, method="eLORETA")

    # Parcellate
    src = mne.read_source_spaces(src_fname)
    labels = mne.read_labels_from_annot(subjects_dir=subjects_dir, subject=subject, parc="aparc")
    ltc = mne.extract_label_time_course(stc, labels, src, mode="pca_flip")
    save_parc_data(ltc, preproc_raw_fname, output_file=f"{parc_dir}/{subject}_run-{run:02d}_parc-raw.fif")
