"""Convert Nottingham files into fif.

"""

import os
from osl_ephys.utils import opm


base_dir = "dog_day_afternoon_OPM"
outdir = "data/raw"

os.makedirs(outdir, exist_ok=True)

for sub in range(1, 11):
    for run in range(1, 3):
        # MEG data
        mat_file = f"{basedir}/sub-{sub:03d}/meg/sub-{sub:03d}_task-movie_run-{run:03d}_meg.mat"
        tsv_file = f"{basedir}/sub-{sub:03d}/meg/sub-{sub:03d}_task-movie_run-{run:03d}_channels.tsv_new"
        out_fif_file = f"{outdir}/sub-{sub:03d}_run-{run:03d}_raw.fif"

        # sMRI data
        smri_file = f"{basedir}/sub-{sub:03d}/anat/sub-{sub:03d}.nii"
        out_smri_file = f"{outdir}/sub-{sub:03d}_T1w.nii"

        # Convert files
        opm.convert_notts(mat_file, smri_file, tsv_file, out_fif_file, out_smri_file)
