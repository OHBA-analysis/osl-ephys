"""Coregister on FreeSurfer surfaces instead of RHINO.

The alternative to 2_coregister.py: recon-all once per subject (hours),
the watershed BEM and MNE's coregistration. Needs FreeSurfer; set
FREESURFER_HOME or call setup_freesurfer.
"""

from osl_ephys.pipeline import run_batch
from osl_ephys.source_recon import setup_freesurfer
from sessions import sessions

setup_freesurfer("/Applications/freesurfer/7.4.1")

config = """
coreg:
  - recon_all: {}
  - make_watershed_bem: {}
  - extract_fiducials_and_headshape: {}
  - coregister: {n_init: 20}
"""

if __name__ == "__main__":
    run_batch(
        config, sessions, "data", n_workers=4, surface_extraction_method="freesurfer"
    )
