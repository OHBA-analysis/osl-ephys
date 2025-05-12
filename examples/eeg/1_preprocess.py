"""Preprocessing.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import mne
import numpy as np
from osl_ephys import preprocessing

config = """
    preproc:
    - resample: {sfreq: 250}
    - bad_segments: {segment_len: 500, picks: eeg}
    - bad_segments: {segment_len: 500, picks: eeg, mode: diff}
    - set_eeg_reference: {projection: true}
"""

preprocessing.run_proc_chain(
    config,
    infile="data/sub-001_clean-raw.fif",
    subject="sub-001",
    outdir="output",
    overwrite=True,
)
