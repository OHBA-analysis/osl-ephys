"""Source reconstruction and parcellation.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl_ephys import source_recon

config = """
    source_recon:
    - forward_model:
        model: Triple Layer
        eeg: True
    - beamform_and_parcellate:
        freq_range: [1, 40]
        chantypes: eeg
        rank: {eeg: 40}
        parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
        method: spatial_basis
        orthogonalisation: symmetric
"""

source_recon.run_src_chain(
    config,
    subject="sub-001",
    outdir="output",
)
