import os
import numpy as np
from dask.distributed import Client
from osl import source_recon, utils

source_recon.setup_fsl("~/fsl") # FSL needs to be installed

def fix_headshape_points(outdir, subject):
    filenames = source_recon.rhino.get_coreg_filenames(outdir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    # Remove headshape points on the nose
    remove = np.logical_and(hs[1] > max(lpa[1], rpa[1]), hs[2] < nas[2])
    hs = hs[:, ~remove]

    # Overwrite headshape file
    utils.logger.log_or_print(f"overwritting {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=16, threads_per_worker=1)

    config = """
        source_recon:
        - extract_polhemus_from_info: {}
        - fix_headshape_points: {}
        - compute_surfaces:
            include_nose: False
        - coregister:
            use_nose: False
            use_headshape: True
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

    basedir = "ds117"
    proc_dir = os.path.join(basedir, "processed")

    # Define inputs
    subjects = [f"sub{i+1:03d}-run{j+1:02d}" for i in range(19) for j in range(6)]
    preproc_files = sorted(utils.Study(os.path.join(proc_dir, "sub{sub_id}-    run{run_id}/sub{sub_id}-run{run_id}_preproc-raw.fif")).get())
    smri_files = np.concatenate([[smri_file]*6 for smri_file in sorted(utils.Study(os.path.join(basedir, "sub{sub_id}/anatomy/highres001.nii.gz"))).get()])

    # Run source batch
    source_recon.run_src_batch(
        config,
        outdir=proc_dir,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        extra_funcs=[fix_headshape_points],
        dask_client=True,
        random_seed=1392754308,
    )

