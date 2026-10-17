"""MaxFilter the raw recordings.

This script needs a computer with a MaxFilter licence. The maxfiltered
files go to data/maxfilter and the preprocessing script reads them from
there. cal and ctc are the fine-calibration and cross-talk files of the
scanner (at OHBA, scanner: Neo sets them, see examples/oxford). The -trans
option is not used because it moves the head, which affects the
coregistration.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
maxfilter:
  - maxfilter:
      maxpath: /neuro/bin/util/maxfilter
      cal: /neuro/databases/sss/sss_cal.dat
      ctc: /neuro/databases/ctc/ct_sparse.fif
      mode: multistage
      tsss: true
      headpos: true
      movecomp: true
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=2)
