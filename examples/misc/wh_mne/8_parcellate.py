"""Parcellation.

"""

import os
import mne
import numpy as np

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
    

subjects_dir = "../data"
subject = "sub-01"

parc_dir = f"{subjects_dir}/parc-data"

os.makedirs(parc_dir, exist_ok=True)

preproc_raw_fname = f"{subjects_dir}/preproc-data/{subject}-preproc-raw.fif"
#src_fname = f"{subjects_dir}/{subject}/space-src.fif"
src_fname = f"{subjects_dir}/morph-maps/fsaverage-src.fif"
stc_fname = f"{subjects_dir}/stc-data/{subject}-lh.stc"

# https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation
#labels = mne.read_labels_from_annot(subjects_dir=subjects_dir, subject=subject, parc="aparc")
labels = mne.read_labels_from_annot(subjects_dir=subjects_dir, subject="fsaverage", parc="aparc")
labels = [l for l in labels if "unknown" not in l.name]

#for i, l in enumerate(labels):
#    print(i, l)

#stc = mne.read_source_estimate(stc_fname, subject=subject)
stc = mne.read_source_estimate(stc_fname, subject="fsaverage")

src = mne.read_source_spaces(src_fname)

ltc = mne.extract_label_time_course(stc, labels, src, mode="pca_flip")

save_parc_data(ltc, preproc_raw_fname, output_file=f"{parc_dir}/{subject}-parc-raw.fif")
