"""Preprocess the recordings.

The ICA components flagged by the EOG and ECG correlation are labelled but
not removed (apply: false), so the labels can be refined by hand before
they are applied. In the paper the following sessions were refined with
osl-ephys ica review data <id> --reject all:

    sub008_run-03, sub019_run-01 to sub019_run-06, sub010_run-05

and the rest cleaned with osl-ephys ica apply data.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
meta:
  event_codes:
    famous/first: 5
    famous/immediate: 6
    famous/last: 7
    unfamiliar/first: 13
    unfamiliar/immediate: 14
    unfamiliar/last: 15
    scrambled/first: 17
    scrambled/immediate: 18
    scrambled/last: 19
preproc:
  - find_events: {min_duration: 0.005}
  - set_channel_types: {EEG061: eog, EEG062: eog, EEG063: ecg}
  - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
  - notch_filter: {freqs: 50 100}
  - resample: {sfreq: 250}
  - bad_segments: {segment_len: 500, picks: mag}
  - bad_segments: {segment_len: 500, picks: grad}
  - bad_segments: {segment_len: 500, picks: mag, mode: diff}
  - bad_segments: {segment_len: 500, picks: grad, mode: diff}
  - bad_channels: {picks: mag, significance_level: 0.1}
  - bad_channels: {picks: grad, significance_level: 0.1}
  - ica_raw: {picks: meg, n_components: 40}
  - ica_autoreject: {ecgmethod: correlation, eogmethod: correlation, eogthreshold: 0.35, apply: false}
  - interpolate_bads: {reset_bads: false}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data", n_workers=16)
