import os
from dask.distributed import Client

import osl


if __name__ == "__main__": 
  client = Client(n_workers=16, threads_per_worker=1) # specify to enable parallel processing
  basedir = "ds117"

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
      - ica_autoreject: {picks: meg, ecgmethod: correlation, eogmethod: correlation,
            eogthreshold: 0.35, apply: False}
      - interpolate_bads: {reset_bads: False}
      """

  # Study utils enables selection of existing paths using various wild cards
  study = osl.utils.Study(os.path.join(basedir, "sub{sub_id}/MEG/run_{run_id}_raw.fif"))
  inputs = sorted(study.get())
  
  # specify session names and output directory
  subjects = [f"sub{i+1:03d}-run{j+1:02d}" for i in range(19) for j in range(6)]
  outdir = os.path.join(basedir, "processed")

  osl.preprocessing.run_proc_batch(
    config,
    inputs,
    subjects,
    outdir,
    dask_client=True,
    random_seed=2280431064,
  )
