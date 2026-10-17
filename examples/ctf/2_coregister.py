"""Coregister with the headshape points of the Polhemus .pos file."""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
coreg:
  - extract_surfaces: {include_nose: false}
  - extract_fiducials_and_headshape_from_pos: {}
  - remove_stray_headshape_points: {}
  - coregister: {use_nose: false, use_headshape: true}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=4)
