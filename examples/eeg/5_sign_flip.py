"""Performs sign flipping.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from glob import glob
from dask.distributed import Client

from osl_ephys import utils
from osl_ephys.source_recon import find_template_subject, run_src_batch

outdir = "output"

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=4, threads_per_worker=1)

    subjects = []
    for path in sorted(glob(f"{outdir}/*/parc/lcmv-parc-raw.fif")):
        subject = path.split("/")[-3]
        subjects.append(subject)

    template = find_template_subject(
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

    run_src_batch(
        config,
        outdir=outdir,
        subjects=subjects,
        dask_client=True,
    )
