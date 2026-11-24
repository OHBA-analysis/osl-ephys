"""Forward model, beamforming and parcellation."""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
source_recon:
  - forward_model: {model: Single Layer}
  - beamform: {chantypes: mag, rank: {mag: 100}, frequency_range: [1, 45]}
parcellation:
  - parcellate:
      parcellation_file: atlas-AAL_nparc-78_space-MNI_res-8x8x8.nii.gz
      orthogonalisation: symmetric
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=4)
