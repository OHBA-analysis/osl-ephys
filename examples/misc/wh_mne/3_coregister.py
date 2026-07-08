"""Coregistration.

https://mne.tools/stable/auto_tutorials/forward/25_automated_coreg.html
"""

import os
import numpy as np

import mne
from mne.coreg import Coregistration
from mne.io import read_info

def save_coreg_html(filename):
    fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
    print("Saving", filename)
    fig.plotter.export_html(filename)

raw_dir = "../data/raw"

subjects_dir = "../data"
subject = "sub-03"

raw_fname = f"{raw_dir}/{subject}/{subject}_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif"
info = read_info(raw_fname)
plot_kwargs = dict(
    subject=subject,
    subjects_dir=subjects_dir,
    surfaces="head",
    dig=True,
    eeg=[],
    meg="sensors",
    show_axes=True,
    coord_frame="meg",
)
view_kwargs = dict(azimuth=45, elevation=90, distance=0.6, focalpoint=(0.0, 0.0, 0.0))

fiducials = "estimated"  # get fiducials from fsaverage
coreg = Coregistration(info, subject, subjects_dir, fiducials=fiducials)
coreg.fit_fiducials(verbose=True)
#coreg.fit_icp(n_iterations=6, nasion_weight=2.0, verbose=True)
#coreg.omit_head_shape_points(distance=1e-3)

save_coreg_html(f"{subjects_dir}/{subject}/coreg.html")

dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
print(
    f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
    f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
)

mne.write_trans(f"{subjects_dir}/{subject}/coreg-trans.fif", coreg.trans, overwrite=True)
