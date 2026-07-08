"""Manual ICA.

This script was tested using Spyder. We recommend running one cell at time.
"""

# Authors: Oliver Kohl <oliver.kohl@psych.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import matplotlib.pyplot as plt

from osl import preprocessing

get_ipython().run_line_magic("matplotlib", "qt")

def check_ica(raw, ica, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    # Find EOG and ECG correlations
    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ecg_indices, ecg_scores = ica.find_bads_ecg(raw)
    ica_scores = ecg_scores + eog_scores

    # Barplot of ICA component "EOG match" and "ECG match" scores
    ica.plot_scores(ica_scores)
    plt.savefig(f"{save_dir}/correl_plot.png", bbox_inches="tight")
    plt.close()

    # Plot bad components
    ica.plot_components(ica.exclude)
    plt.savefig(f"{save_dir}/bad_components.png", bbox_inches="tight")
    plt.close()

def plot_psd(raw, save_dir):
    raw.compute_psd(fmax=45, n_fft=int(raw.info["sfreq"] * 4)).plot()
    plt.savefig(f"{save_dir}/powspec.png", bbox_inches="tight")
    plt.close()

#%% Setup paths

# Directories
preproc_dir = "data/preproc"
output_dir = "data/preproc_ica"

# Subjects
subjects = ["sub-001", "sub-002"]

# Paths to files
preproc_files = []
output_raw_files = []
output_ica_files = []
report_dirs = []
for subject in subjects:
    preproc_files.append(f"{preproc_dir}/{subject}/{subject}_tsss_preproc_raw.fif")
    output_raw_files.append(f"{output_dir}/{subject}/{subject}_tsss_preproc_raw.fif")
    output_ica_files.append(f"{output_dir}/{subject}/{subject}_tsss_ica.fif")
    report_dirs.append(f"{output_dir}/report/{subject}")

#%% Manual ICA ICA artefact rejection

# Index for the preprocessed data file we want to do ICA for
index = 0
subject = subjects[index]
print("Doing", subject)

# Files for the corresponding index
preproc_file = preproc_files[index]
output_raw_file = output_raw_files[index]
output_ica_file = output_ica_files[index]
report_dir = report_dirs[index]

# Create output directories
os.makedirs(os.path.dirname(output_raw_file), exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# Load preprocessed fif and ICA
dataset = preprocessing.read_dataset(preproc_file, preload=True)
raw = dataset["raw"]
ica = dataset["ica"]

# Mark bad ICA components interactively
preprocessing.plot_ica(ica, raw)

#%% Create figures to check ICA with
check_ica(raw, ica, report_dir)
plt.close("all")

# Apply ICA
print()
raw = ica.apply(raw, exclude=ica.exclude)

print()
print("Removed components:", ica.exclude)
print()

# Plot power spectrum of cleaned data
plot_psd(raw, report_dir)

#%% Save cleaned data
dataset["raw"].save(output_raw_file, overwrite=True)
dataset["ica"].save(output_ica_file, overwrite=True)
