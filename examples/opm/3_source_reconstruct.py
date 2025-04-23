"""Source reconstruction with an LCMV beamformer.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from dask.distributed import Client
from osl_ephys import source_recon, utils

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=4, threads_per_worker=1)

    config = """
        source_recon:
        - forward_model:
            model: Single Layer
        - beamform_and_parcellate:
            freq_range: [1, 45]
            chantypes: mag
            rank: {mag: 100}
            spatial_resolution: 8
            parcellation_file: aal_cortical_merged_8mm_stacked.nii.gz
            method: spatial_basis
            orthogonalisation: symmetric
    """

    subjects = []
    for sub in range(1, 11):
        for run in range(1, 3):
            subjects.append(f"sub-{sub:03d}_run-{run:03d}")

    outdir = "data/preproc"

    source_recon.run_src_batch(
        config,
        outdir=outdir,
        subjects=subjects,
        dask_client=True,
    )
