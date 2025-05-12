"""Coregisteration.

Note, this script does not actually move the EEG electrodes in the fif file.

This script should be used to find the best rescaling for the EEG sensor
positions to match the structural.

The next script, 3_move_sensors.py, moves the sensors.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl_ephys import source_recon

config = """
    source_recon:
    - extract_polhemus_from_info:
        include_eeg_as_headshape: True
        rescale: [0.9, 1, 1]
    - compute_surfaces:
        include_nose: False
    - coregister:
        use_nose: False
        use_headshape: True
"""

source_recon.run_src_chain(
    config,
    smri_file="data/sub-001_T1w.nii.gz",
    subject="sub-001",
    outdir="output",
)
