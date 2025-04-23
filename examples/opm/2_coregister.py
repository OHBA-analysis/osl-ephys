"""Coregistration with RHINO.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from dask.distributed import Client
from osl_ephys import source_recon, utils

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=4, threads_per_worker=1)

    config = """
        source_recon:
        - compute_surfaces:
            include_nose: False
        - coregister:
            use_nose: False
            use_headshape: False
            already_coregistered: True
    """

    subjects, smri_files = [], []
    for sub in range(1, 11):
        for run in range(1, 3):
            subjects.append(f"sub-{sub:03d}_run-{run:03d}")
            smri_files.append(f"../data/raw/sub-{sub:03d}_T1w.nii")

    outdir = "data/preproc"

    source_recon.run_src_batch(
        config,
        outdir=outdir,
        subjects=subjects,
        smri_files=smri_files,
        dask_client=True,
    )
