"""MaxFilter the raw recordings.

This script needs a computer with a MaxFilter licence. scanner: Neo sets
the calibration files of the OHBA TRIUX (recordings since 2019; see
../maxfilter for older recordings). The maxfiltered files go to
data/maxfilter and the preprocessing script reads them from there. The
-trans option is not used because it moves the head, which affects the
coregistration.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

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
