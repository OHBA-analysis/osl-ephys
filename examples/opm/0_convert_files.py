"""Convert the Nottingham MATLAB files to fif, fixing the MRI's sform."""

import os

from osl_ephys.io import opm

base_dir = "dog_day_afternoon_OPM"
outdir = "data/raw"
os.makedirs(outdir, exist_ok=True)

for sub in range(1, 11):
    for run in range(1, 3):
        meg = f"{base_dir}/sub-{sub:03d}/meg/sub-{sub:03d}_task-movie_run-{run:03d}"
        opm.convert_notts(
            f"{meg}_meg.mat",
            f"{base_dir}/sub-{sub:03d}/anat/sub-{sub:03d}.nii",
            f"{meg}_channels.tsv_new",
            f"{outdir}/sub-{sub:03d}_run-{run:03d}_raw.fif",
            f"{outdir}/sub-{sub:03d}_T1w.nii",
        )
