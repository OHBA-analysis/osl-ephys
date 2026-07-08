"""RHINO without any visualisations.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import numpy as np

from osl.source_recon import rhino

# Directories
preproc_dir = "data/preproc"
anat_dir = "data/smri"
coreg_dir = "data/coreg"

# Files ({subject} will be replaced by the name for the subject)
preproc_file = preproc_dir + "/{subject}_tsss_preproc_raw.fif"
smri_file = anat_dir + "/{subject}/anat/{subject}_T1w.nii"

# Subjects to coregister
subjects = ["sub-001", "sub-002"]

# Main RHINO code
for subject in subjects:

    # Input files
    preproc_file_ = preproc_file.format(subject=subject)
    smri_file_ = smri_file.format(subject=subject)

    # Extract fiducials and headshape points
    filenames = rhino.get_coreg_filenames(src_dir, subject)
    rhino.extract_polhemus_from_info(
        fif_file=preproc_file_,
        headshape_outfile=filenames["polhemus_headshape_file"],
        nasion_outfile=filenames["polhemus_nasion_file"],
        rpa_outfile=filenames["polhemus_rpa_file"],
        lpa_outfile=filenames["polhemus_lpa_file"],
    )

    # Clean up headshape points
    rhino.remove_stray_headshape_points(src_dir, subject)

    # Extract surfaces from the structural
    rhino.compute_surfaces(
        smri_file=smri_file_,
        subjects_dir=src_dir,
        subject=subject,
        include_nose=False,
    )

    # Coregistration
    rhino.coreg(
        fif_file=preproc_file_,
        subjects_dir=src_dir,
        subject=subject,
        use_headshape=True,
        use_nose=False,
    )

    # Forward model
    rhino.forward_model(
        subjects_dir=src_dir,
        subject=subject,
        model="Single Layer"
        gridstep=8,
    )
