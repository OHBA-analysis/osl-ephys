"""Performs sign flipping.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from glob import glob
from dask.distributed import Client

from osl_ephys import utils
from osl_ephys.source_recon import find_template_subject, run_src_batch

# Directories
outdir = "data"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    # Subjects to sign flip
    # We create a list by looking for subjects that have a parc/parc-raw.fif file
    subjects = []
    for path in sorted(glob(f"{outdir}/*/parc/parc-raw.fif")):
        subject = path.split("/")[-3]
        subjects.append(subject)

    # Find a good template subject to align other subjects to
    template = find_template_subject(
        outdir, subjects, n_embeddings=15, standardize=True
    )

    # Settings for batch processing
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

    # Setup parallel processing
    #
    # n_workers is the number of CPUs to use,
    # we recommend less than half the total number of CPUs you have
    client = Client(n_workers=4, threads_per_worker=1)

    # Do the sign flipping
    run_src_batch(
        config,
        outdir=outdir,
        subjects=subjects,
        dask_client=True,
    )
