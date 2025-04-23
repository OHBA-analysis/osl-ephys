# Authors: Mats van Es <mats.vanes@psych.ox.ac.uk>

from osl_ephys import source_recon
source_recon.setup_freesurfer('/Applications/freesurfer/7.4.1')

recon_dir = '/Users/matsvanes/osl-dev/output'
subject = 'sub-oxf001'
preproc_file = '/Users/matsvanes/osl-dev/output/sub-oxf001_task-resteyesopen/sub-oxf001_task-resteyesopen_preproc-raw.fif'
smri_file='/Users/matsvanes/osl-dev/smri/sub-oxf001_T1w.nii.gz'

subjects = [subject]
smri_files = [smri_file]
preproc_files = [preproc_file]

# Run FreeSurfer recon-all before running the pipeline
for subject, smri_file in zip(subjects, smri_files):
    source_recon.recon_all(smri_file, recon_dir, subject)
    
config = """
    source_recon:
    - make_watershed_bem: {}
    - coregister:
        nasion_weight: 2.0
    - forward_model:
        forward_model: Single Layer
        source_space: surface
        kwargs: {ico: 4}
        gridstep: 8
    - minimum_norm_and_parcellate:
        source_space: surface
        source_method: eLORETA
        chantypes: [mag, grad]
        rank: {meg: 20}
        depth: 0.8
        loose: 0.2
        reg: 0.1
        pick_ori: None
        parcellation_file: Yeo2011_7Networks_N1000
        reference_brain: fsaverage
        method: pca_flip
        orthogonalisation: symmetric
"""

source_recon.run_src_batch(
    config,
    outdir=recon_dir,
    subjects=[subject],
    preproc_files=preproc_files,
    smri_file=[smri_file],   
    surface_extraction_method='freesurfer'
)