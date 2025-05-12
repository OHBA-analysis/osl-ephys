"""Move EEG sensors and re-do coregistration.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl_ephys import source_recon

config = """
    source_recon:
    - rescale_sensor_positions:
        rescale: [0.9, 1, 1]

    # Redo coregistration
    - extract_polhemus_from_info:
        include_eeg_as_headshape: True
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
