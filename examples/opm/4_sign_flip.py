"""Sign flipping.

Note, this script is only needed if you're training a dynamic network
model (e.g. the HMM) using the time-delay embedded (TDE) approach.

You can skip this if you're training the HMM on amplitude envelope data
or calculating sign-invariant quantities such as amplitude envelope
correlations or power.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from dask.distributed import Client
from osl_ephys import source_recon, utils

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=4, threads_per_worker=1)

    subjects = []
    for sub in range(1, 11):
        for run in range(1, 3):
            subjects.append(f"sub-{sub:03d}_run-{run:03d}")

    outdir = "data/preproc"

    # Find a good template subject to align other subjects to
    template = source_recon.find_template_subject(
        outdir, subjects, n_embeddings=15, standardize=True
    )

    config = f"""
        source_recon:
        - fix_sign_ambiguity:
            template: {template}
            n_embeddings: 15
            standardize: True
            n_init: 3
            n_iter: 2500
            max_flips: 20
    """

    source_recon.run_src_batch(
        config,
        outdir=outdir,
        subjects=subjects,
        dask_client=True,
    )
