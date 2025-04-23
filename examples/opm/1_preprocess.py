"""Preprocess OPM data.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>


from dask.distributed import Client
from osl_ephys import preprocessing, utils

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=4, threads_per_worker=1)

    config = """
        preproc:
        - resample: {sfreq: 250}
        - filter: {l_freq: 1, h_freq: 45, method: iir, iir_params: {order: 5, ftype: butter}}
        - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
        - bad_channels: {picks: mag, significance_level: 0.4}
    """

    subjects, inputs = [], []
    for sub in range(1, 11):
        for run in range(1, 3):
            subjects.append(f"sub-{sub:03d}_run-{run:03d}")
            inputs.append(f"../data/raw/sub-{sub:03d}_run-{run:03d}_raw.fif")

    outdir = "data/preproc"

    dataset = preprocessing.run_proc_batch(
        config,
        inputs,
        subjects=subjects,
        outdir=outdir,
        overwrite=True,
        dask_client=True,
    )
