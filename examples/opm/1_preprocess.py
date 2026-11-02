"""Preprocess the recordings.

The adaptive multipole model (amm) separates the brain's field from the
interference using the sensor geometry; the GESD detectors then mark the
bad segments and channels that remain.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
preproc:
  - resample: {sfreq: 250}
  - filter: {l_freq: 1, h_freq: 45, method: iir, iir_params: {order: 5, ftype: butter}}
  - amm: {li: 9, le: 2}
  - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
  - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
  - bad_channels: {picks: mag, significance_level: 0.4}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=4)
