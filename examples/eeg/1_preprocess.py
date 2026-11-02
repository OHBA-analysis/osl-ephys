"""Preprocess the recordings."""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
preproc:
  - set_channel_types: {VEOG: eog, HEOG: eog, ECG: ecg}
  - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
  - notch_filter: {freqs: 50 100}
  - resample: {sfreq: 250}
  - bad_segments: {segment_len: 500, picks: eeg}
  - bad_segments: {segment_len: 500, picks: eeg, mode: diff}
  - bad_channels: {picks: eeg}
  - ica_raw: {picks: eeg, n_components: 30}
  - ica_autoreject: {ecgmethod: correlation, eogthreshold: auto}
  - interpolate_bads: {}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=4)
