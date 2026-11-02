"""Forward model, beamforming and parcellation.

EEG needs the three-layer head model.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
source_recon:
  - forward_model: {model: Triple Layer, eeg: true}
  - beamform: {chantypes: eeg, rank: {eeg: 40}, frequency_range: [1, 40]}
parcellation:
  - parcellate:
      parcellation_file: atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz
      orthogonalisation: symmetric
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=4)
