"""Preprocess the maxfiltered recordings.

The sessions run in parallel, n_workers at a time; use fewer than half the
CPUs of the computer. Each session's log is in data/logs and the QC report
in data/qc/report.html.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
maxfilter:
  - maxfilter: {maxpath: /neuro/bin/util/maxfilter, scanner: Neo, mode: multistage, tsss: true, headpos: true, movecomp: true}
preproc:
  - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
  - notch_filter: {freqs: 50 100}
  - resample: {sfreq: 250}
  - bad_segments: {segment_len: 500, picks: mag}
  - bad_segments: {segment_len: 500, picks: grad}
  - bad_segments: {segment_len: 500, picks: mag, mode: diff}
  - bad_segments: {segment_len: 500, picks: grad, mode: diff}
  - bad_channels: {picks: mag}
  - bad_channels: {picks: grad}
  - ica_raw: {picks: meg, n_components: 40}
  - ica_autoreject: {ecgmethod: correlation, eogthreshold: auto}
  - interpolate_bads: {}
"""

if __name__ == "__main__":
    # The maxfilter section is skipped for sessions whose maxfiltered file exists
    run_batch(config, sessions, "data", n_workers=4)
