"""Signal-space projection (SSP) instead of ICA for the eye and heart
artefacts.

One projector per sensor type is computed from the ECG and from the EOG
epochs of the preprocessed data, applied, and the cleaned data written back
over data/preproc/<id>_preproc-raw.fif. Figures of the projectors and the
power spectrum go to data/ssp. Increase n_proj to 2 or 3 for the ECG of a
session whose heartbeat is still visible afterwards.
"""

import os

import matplotlib.pyplot as plt
import mne

from osl_ephys.report import headless
from sessions import sessions

n_proj = 1
os.makedirs("data/ssp", exist_ok=True)

for session in sessions:
    id = session["id"]
    preproc_file = f"data/preproc/{id}_preproc-raw.fif"
    raw = mne.io.read_raw_fif(preproc_file, preload=True)
    raw.pick(["meg", "ecg", "eog", "emg"])

    # The projectors are computed without the bad channels
    good = raw.copy()
    good.info["bads"] = []

    ecg_epochs = mne.preprocessing.create_ecg_epochs(good, picks="all").average(
        picks="all"
    )
    ecg_projs, _ = mne.preprocessing.compute_proj_ecg(
        good, n_grad=n_proj, n_mag=n_proj, n_eeg=0, no_proj=True, reject=None
    )
    eog_epochs = mne.preprocessing.create_eog_epochs(good, picks="all").average(
        picks="all"
    )
    eog_projs, _ = mne.preprocessing.compute_proj_eog(
        good, n_grad=n_proj, n_mag=n_proj, n_eeg=0, no_proj=True, reject=None
    )

    raw.add_proj(ecg_projs + eog_projs)
    raw.apply_proj()
    raw.save(preproc_file, overwrite=True)

    with headless():
        if ecg_projs:
            mne.viz.plot_projs_joint(ecg_projs, ecg_epochs, show=False)
            plt.savefig(f"data/ssp/{id}_ecg.png", bbox_inches="tight")
            plt.close()
        if eog_projs:
            mne.viz.plot_projs_joint(eog_projs, eog_epochs, show=False)
            plt.savefig(f"data/ssp/{id}_eog.png", bbox_inches="tight")
            plt.close()
        raw.compute_psd(fmax=45, n_fft=int(raw.info["sfreq"] * 4)).plot(show=False)
        plt.savefig(f"data/ssp/{id}_psd.png", bbox_inches="tight")
        plt.close()
