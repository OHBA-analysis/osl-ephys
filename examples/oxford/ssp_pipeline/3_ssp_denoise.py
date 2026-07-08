"""Signal-space projection (SSP) denoising.

Increase n_proj (line 55) for the ECG to 2 or 3 for particular subjects
may improve the denoising.
"""

# Authors: Mats van Es <mats.vanes@psych.ox.ac.uk>
#          Oliver Kohl <oliver.kohl@psych.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import mne
import matplotlib.pyplot as plt

from osl import preprocessing


# Directories
preproc_dir = "data/preproc"
ssp_preproc_dir = "data/preproc_ssp"
report_dir = "data/preproc_ssp/report"

os.makedirs(ssp_preproc_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# Subjects
subjects = ["sub-001", "sub-002"]

# Paths to files
preproc_files = []
ssp_preproc_files = []
for subject in subjects:
    preproc_files.append(f"{preproc_dir}/{subject}/{subject}_preproc_raw.fif")
    ssp_preproc_files.append(f"{ssp_preproc_dir}/{subject}/{subject}_preproc_raw.fif")

for index in range(len(preproc_files)):
    subject = subjects[index]
    preproc_file = preproc_files[index]
    output_raw_file = ssp_preproc_files[index]

    # Make output directory
    os.makedirs(os.path.dirname(output_raw_file), exist_ok=True)

    # Load preprocessed fif and ICA
    dataset = preprocessing.read_dataset(preproc_file, preload=True)
    raw = dataset["raw"]

    # Only keep MEG, ECG, EOG, EMG
    raw = raw.pick_types(meg=True, ecg=True, eog=True, emg=True)

    # Create a Raw object without any channels marked as bad
    raw_no_bad_channels = raw.copy()
    raw_no_bad_channels.load_bad_channels()

    #  Calculate SSP using ECG
    n_proj = 1
    ecg_epochs = mne.preprocessing.create_ecg_epochs(
        raw_no_bad_channels, picks="all"
    ).average(picks="all")
    ecg_projs, events = mne.preprocessing.compute_proj_ecg(
        raw_no_bad_channels,
        n_grad=n_proj,
        n_mag=n_proj,
        n_eeg=0,
        no_proj=True,
        reject=None,
        n_jobs=6,
    )

    # Add ECG SSPs to Raw object
    raw_ssp = raw.copy()
    raw_ssp.add_proj(ecg_projs.copy())

    # Calculate SSP using EOG
    n_proj = 1
    eog_epochs = mne.preprocessing.create_eog_epochs(
        raw_no_bad_channels, picks="all"
    ).average()
    eog_projs, events = mne.preprocessing.compute_proj_eog(
        raw_no_bad_channels,
        n_grad=n_proj,
        n_mag=n_proj,
        n_eeg=0,
        no_proj=True,
        reject=None,
        n_jobs=6,
    )

    # Add EOG SSPs to Raw object
    raw_ssp.add_proj(eog_projs.copy())

    # Apply SSPs
    raw_ssp.apply_proj()

    # Plot power spectrum of cleaned data
    raw_ssp.plot_psd(fmax=45, n_fft=int(raw.info["sfreq"] * 4))
    plt.savefig(f"{report_dir}/psd_{subject}.png", bbox_inches="tight")
    plt.close()

    if len(ecg_projs) > 0:
        fig = mne.viz.plot_projs_joint(ecg_projs, ecg_epochs, show=False)
        plt.savefig(f"{report_dir}/proj_ecg_{subject}.png", bbox_inches="tight")
        plt.close()

    if len(eog_projs) > 0:
        fig = mne.viz.plot_projs_joint(eog_projs, eog_epochs, show=False)
        plt.savefig(f"{report_dir}/proj_eog_{subject}.png", bbox_inches="tight")
        plt.close()

    # Save cleaned data
    raw_ssp.save(output_raw_file, overwrite=True)
