"""Source reconstruction: the LCMV beamformer on the RHINO forward model, and
the FreeSurfer route (coregistration, forward model and minimum norm with MNE).

Each stage reads and writes the files held by a :class:`~osl_ephys.session.Session`.
"""

from .freesurfer import (
    setup_freesurfer,
    check_freesurfer,
    recon_all,
    make_watershed_bem,
    fsaverage_source_space,
)
