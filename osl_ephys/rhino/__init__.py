"""RHINO: coregistration and forward modelling from a structural MRI with FSL.

The steps read and write the files of a :class:`~osl_ephys.session.Session`:
surface extraction, fiducial and headshape extraction, coregistration and
the forward model on the MNI grid.
"""

from osl_ephys.rhino.fsl import (
    setup_fsl,
    check_fsl,
    fsleyes,
    fsleyes_overlay,
)
from osl_ephys.rhino.surfaces import (
    align_smri_to_mni,
    extract_surfaces,
    plot_surfaces,
    repair_bem_surfaces,
)
from osl_ephys.rhino.headshape import (
    extract_fiducials_and_headshape_from_fif,
    extract_fiducials_and_headshape_from_pos,
    extract_fiducials_and_headshape_from_elc,
    remove_stray_headshape_points,
    delete_headshape_points,
)
from osl_ephys.rhino.coregistration import (
    scale_surfaces_to_headshape,
    save_coregistration_files,
    coregister_head_and_mri,
    coreg_error,
    coreg_rms,
    plot_coregistration,
)
from osl_ephys.rhino.forward import (
    forward_model,
)
