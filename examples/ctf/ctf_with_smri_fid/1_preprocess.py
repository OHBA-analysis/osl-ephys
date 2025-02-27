"""Preprocessing.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

from osl_ephys import preprocessing

# Settings
config = """
    preproc:
    - filter: {l_freq: 1, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
    - notch_filter: {freqs: 50 100}
    - resample: {sfreq: 250}
"""

# Create a list of paths to files to preprocess
inputs = ["data/raw/mg04938_BrainampDBS_20170504_01_raw.fif"]

# Subject IDs
subjects = ["LN_VTA2"]

# Directory to save output to
outdir = "data"

# Do preprocessing
preprocessing.run_proc_batch(
    config,
    inputs,
    subjects=subjects,
    outdir=outdir,
    overwrite=True,
)
