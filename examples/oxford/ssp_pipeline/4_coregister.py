"""Coregisteration.

The scripts was first run for all subjects (with n_init=1). Then for subjects
whose coregistration looked a bit off we re-run this script just for that
particular subject with a higher n_init.

Note, these scripts do not include/use the nose in the coregistration.
If you want to use the nose you need to change the config to include the nose
and you may not want to call the remove_stray_headshape_points function.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np
from dask.distributed import Client

from osl import source_recon, utils

# Directories
preproc_dir = "data/preproc_ssp"
anat_dir = "data/smri"
coreg_dir = "data/coreg"

# Files ({subject} will be replaced by the name for the subject)
preproc_file = preproc_dir + "/{subject}_tsss_preproc_raw.fif"
smri_file = anat_dir + "/{subject}/anat/{subject}_T1w.nii"

# Subjects to coregister
subjects = ["sub-001", "sub-002"]

# Settings
config = """
    source_recon:
    - extract_polhemus_from_info: {}
    - remove_stray_headshape_points: {}
    - compute_surfaces:
        include_nose: False
    - coregister:
        use_nose: False
        use_headshape: True
        #n_init: 50
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Setup files
    preproc_files = []
    smri_files = []
    for subject in subjects:
        preproc_files.append(preproc_file.format(subject=subject))
        smri_files.append(smri_file.format(subject=subject))

    # Setup parallel processing
    #
    # n_workers is the number of CPUs to use,
    # we recommend less than half the total number of CPUs you have
    client = Client(n_workers=4, threads_per_worker=1)

    # Run coregistration
    source_recon.run_src_batch(
        config,
        src_dir=coreg_dir,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        dask_client=True,
    )
