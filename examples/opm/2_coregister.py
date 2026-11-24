"""Surfaces from the MRI, and the coregistration files of an already
coregistered recording."""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
coreg:
  - extract_surfaces: {include_nose: false}
  - coregister: {already_coregistered: true}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=4)
