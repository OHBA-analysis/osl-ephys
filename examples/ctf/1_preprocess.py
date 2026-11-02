"""Preprocess the recordings."""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
preproc:
  - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
  - notch_filter: {freqs: 50 100 150}
  - resample: {sfreq: 250}
  - bad_segments: {segment_len: 500, picks: mag, ref_meg: false}
  - bad_segments: {segment_len: 500, picks: mag, ref_meg: false, mode: diff}
  - bad_channels: {picks: mag, ref_meg: false}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=4)
