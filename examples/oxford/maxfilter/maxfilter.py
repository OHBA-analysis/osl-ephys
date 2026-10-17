"""MaxFilter recordings from the OHBA scanner.

This script needs a computer with a MaxFilter licence. See the README for
the scanner option. The maxfiltered files go to data/maxfilter.
"""

from osl_ephys.pipeline import run_batch

sessions = [
    {"id": "sub-001_task-rest", "raw_file": "raw/sub-001_task-rest.fif"},
    {"id": "sub-002_task-rest", "raw_file": "raw/sub-002_task-rest.fif"},
]

config = """
maxfilter:
  - maxfilter:
      maxpath: /neuro/bin/util/maxfilter
      scanner: Neo
      mode: multistage
      tsss: true
      headpos: true
      movecomp: true
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=2)
