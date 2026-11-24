"""Coregister with the electrodes as headshape points.

Without digitised electrode positions the montage's template positions are
used as the headshape. Check in the coregistration tab of the report that
the electrodes sit on the scalp.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
coreg:
  - extract_surfaces: {include_nose: false}
  - extract_fiducials_and_headshape: {include_eeg_as_headshape: true}
  - coregister: {use_nose: false, use_headshape: true}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=4)
