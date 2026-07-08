"""Plotting.

"""

import mne
import numpy as np
import matplotlib.pyplot as plt

plot_sensor_psd = False
plot_parc_psd = False
plot_parc = False
plot_alpha_activity = False
plot_stc_movie = True

def standardize(data):
    data -= np.mean(data, axis=-1, keepdims=True)
    data /= np.std(data, axis=-1, keepdims=True)
    return data

if plot_sensor_psd:
    raw = mne.io.read_raw_fif("../data/preproc-data/sub-01-preproc-raw.fif", preload=True)
    fig = raw.plot_psd(fmin=0.5, fmax=45, show=False)
    fig.savefig("plots/sensor-psd.png")

if plot_parc_psd:
    from scipy.signal import welch

    raw = mne.io.read_raw_fif("../data/parc-data/sub-01-parc-raw.fif", preload=True)
    raw.apply_function(standardize, picks="misc")
    data = raw.get_data(picks="misc", reject_by_annotation="omit")
    fs = raw.info["sfreq"]

    fig, ax = plt.subplots()
    for x in data:
        f, p = welch(x, fs=fs, nperseg=2 * fs)
        ax.plot(f, p)
    ax.set_xlim(0, 45)
    filename = "plots/parc-psd.png"
    plt.savefig(filename)

if plot_parc:
    #parcels = [23]
    parcels = range(68)

    brain = mne.viz.Brain(
        #subject="sub-01",
        subject="fsaverage",
        hemi="both",
        subjects_dir="../data",
        cortex="low_contrast",
        background="white",
    )

    #labels = mne.read_labels_from_annot(subject="sub-01", parc="aparc", subjects_dir="../data")
    labels = mne.read_labels_from_annot(subject="fsaverage", parc="aparc", subjects_dir="../data")

    for p in parcels:
        brain.add_label(labels[p], alpha=0.9)

    brain.save_image("plots/parc.png")

if plot_alpha_activity:
    raw = mne.io.read_raw_fif("../data/parc-data/sub-01-parc-raw.fif", preload=True)
    raw.apply_function(standardize, picks="misc")
    raw = raw.filter(l_freq=8, h_freq=12, picks="misc")
    data = raw.get_data(picks="misc", reject_by_annotation="omit")

    parcel_data = np.std(data, axis=-1, keepdims=True)

    #labels = mne.read_labels_from_annot(subject="sub-01", parc="aparc", subjects_dir="../data")
    labels = mne.read_labels_from_annot(subject="fsaverage", parc="aparc", subjects_dir="../data")
    labels = [l for l in labels if "unknown" not in l.name]

    #stc = mne.labels_to_stc(labels, parcel_data, subject="sub-01")
    stc = mne.labels_to_stc(labels, parcel_data, subject="fsaverage")

    brain = stc.plot(
        subject="fsaverage",
        subjects_dir="../data",
        hemi="both",
        surface="inflated",
    )

    brain.save_image("plots/test.png")

if plot_stc_movie:
    stc = mne.read_source_estimate("../data/stc-data/sub-01-epo-lh.stc", subject="fsaverage")

    brain = stc.plot(
        subject="fsaverage",
        subjects_dir="../data",
        hemi="both",
        surface="inflated",
    )

    brain.save_movie(
        filename="plots/evoked.mov", 
        tmin=stc.times[0],
        tmax=stc.times[-1],
        framerate=10,
    )
