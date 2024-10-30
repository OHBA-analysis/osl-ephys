
import os
from glob import glob
from dask.distributed import Client

from osl import source_recon, utils

source_recon.setup_fsl("~/fsl")

# Directory containing source reconstructed data
proc_dir = "ds117/processed"
src_files = sorted(utils.Study(os.path.join(proc_dir, 
"sub{sub_id}-run{run_id}/parc/parc-raw.fif")).get())

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")

    subjects = [f"sub{i+1:03d}-run{j+1:02d}" for i in range(19) for j in range(6)]
         
    # Find a good template subject to match others to
    template = source_recon.find_template_subject(
        proc_dir, subjects, n_embeddings=15, standardize=True,
    )

    # Settings
    config = f"""
        source_recon:
        - fix_sign_ambiguity:
            template: {template}
            n_embeddings: 15
            standardize: True
            n_init: 3
            n_iter: 3000
            max_flips: 20
    """

    # Setup parallel processing
    client = Client(n_workers=16, threads_per_worker=1)

    # Run sign flipping
    source_recon.run_src_batch(config, proc_dir, subjects, dask_client=True, random_seed=3116145039)





