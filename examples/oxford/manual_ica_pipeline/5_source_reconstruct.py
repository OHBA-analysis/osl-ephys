"""Source reconstruction.

This includes beamforming, parcellation and orthogonalisation.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
from dask.distributed import Client

from osl import source_recon, utils

# Directories
preproc_dir = "data/preproc_ica"
coreg_dir = "data/coreg"
src_dir = "data/src"

# Files
preproc_file = preproc_dir + "/{subject}_tsss_preproc_raw.fif"  # {subject} will be replaced by the subject name

# Subjects to do
subjects = ["sub-001", "sub-002"]

# Settings
config = """
    source_recon:
    - forward_model:
        model: Single Layer
    - beamform_and_parcellate:
        freq_range: [1, 45]
        chantypes: [mag, grad]
        rank: {meg: 60}
        parcellation_file: Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Copy directory containing the coregistration
    if not os.path.exists(src_dir):
        cmd = f"cp -r {coreg_dir} {src_dir}"
        print(cmd)
        os.system(cmd)

    # Get paths to files
    preproc_files = []
    for subject in subjects:
        preproc_files.append(preproc_file.format(subject=subject))

    # Setup parallel processing
    #
    # n_workers is the number of CPUs to use,
    # we recommend less than half the total number of CPUs you have
    client = Client(n_workers=4, threads_per_worker=1)

    # Source reconstruction
    source_recon.run_src_batch(
        config,
        src_dir=src_dir,
        subjects=subjects,
        preproc_files=preproc_files,
        dask_client=True,
    )
