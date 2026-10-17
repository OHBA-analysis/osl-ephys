"""Forward model, beamforming and parcellation.

The parcel time courses are data/<id>/<id>_parc-raw.fif, with symmetric
orthogonalisation to reduce the spatial leakage.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
source_recon:
  - forward_model: {model: Single Layer}
  - beamform: {chantypes: [mag, grad], rank: {meg: 60}, frequency_range: [1, 45]}
parcellation:
  - parcellate:
      parcellation_file: atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz
      orthogonalisation: symmetric
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=16)
