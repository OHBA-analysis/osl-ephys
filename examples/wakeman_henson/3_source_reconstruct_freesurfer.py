"""Surface forward model, minimum norm and the aparc parcellation.

The alternative to 3_source_reconstruct.py after 2_coregister_freesurfer.py:
eLORETA on a surface source space, with one time course per aparc label.
"""

from osl_ephys.pipeline import run_batch
from osl_ephys.source_recon import setup_freesurfer
from sessions import sessions

setup_freesurfer("/Applications/freesurfer/7.4.1")

config = """
source_recon:
  - forward_model: {spacing: oct6, ico: 5}
  - minimum_norm: {chantypes: [mag, grad], rank: {meg: 60}, loose: 0.2, depth: 0.8}
parcellation:
  - parcellate:
      source_method: minimum_norm
      inverse_method: eLORETA
      lambda2: 0.1
      parcellation_file: aparc
      method: pca_flip
"""

if __name__ == "__main__":
    run_batch(
        config, sessions, "data", n_workers=4, surface_extraction_method="freesurfer"
    )
