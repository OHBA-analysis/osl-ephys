"""Coregister the head to the MRI.

The surfaces are extracted once per subject (extract_surfaces runs on the
first session of each subject) and the coregistration once per session.
The nose is not used; to use it, extract the surfaces with include_nose and
coregister with use_nose, and do not remove the stray headshape points.

Run it for every session first (n_init: 1), then rerun the sessions whose
coregistration looks off in the report with a higher n_init.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
coreg:
  - extract_surfaces: {include_nose: false}
  - extract_fiducials_and_headshape: {}
  - remove_stray_headshape_points: {}
  - coregister: {use_nose: false, use_headshape: true, n_init: 1}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=4)
