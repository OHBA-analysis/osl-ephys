"""The preproc steps, run as configs on each modality's test recording."""

import mne
import numpy as np
import pytest

from osl_ephys.pipeline import run_chain

ELEKTA = """
meta:
  event_codes: {one: 1, two: 2}
preproc:
  - crop: {tmax: 16}
  - filter: {l_freq: 1, h_freq: 45}
  - notch_filter: {freqs: 50}
  - resample: {sfreq: 100}
  - bad_channels: {picks: mag}
  - bad_channels: {picks: grad}
  - bad_segments: {segment_len: 100, picks: mag}
  - bad_segments: {segment_len: 100, picks: grad, mode: diff}
  - interpolate_bads: {}
  - annotate_muscle_zscore: {ch_type: mag, threshold: 4, filter_freq: [30, 45]}
  - find_events: {min_duration: 0.005}
  - epochs: {tmin: -0.2, tmax: 0.5, picks: [meg, eog, ecg]}
  - drop_bad_epochs: {picks: mag}
  - apply_baseline: {baseline: "-0.2 0"}
  - tfr_multitaper: {freqs: "5 20 4", n_cycles: 2, return_itc: true}
  - ica_raw: {picks: meg, n_components: 5, random_state: 0, max_iter: 200}
  - ica_autoreject: {ecgmethod: correlation, apply: true}
  - ica_ecg_eog_correlation: {picks: meg, n_components: 5, ecg_method: correlation, max_iter: 200}
"""

CTF = """
preproc:
  - filter: {l_freq: 1, h_freq: 45}
  - bad_channels: {picks: mag, ref_meg: false}
  - bad_segments: {segment_len: 250, picks: mag, ref_meg: false, significance_level: 0.3}
  - pick_types: {meg: true, ref_meg: false, stim: true}
"""

EEG = """
preproc:
  - crop: {tmax: 20}
  - filter: {l_freq: 1, h_freq: 45}
  - bad_channels: {picks: eeg}
  - bad_segments: {segment_len: 250, picks: eeg}
  - set_channel_types: {Fp1: eog}
  - pick: {picks: [eeg, eog]}
"""

OPM = """
preproc:
  - crop: {tmax: 5}
  - decimate_headshape_points: {decimate_amount: 0.02}
  - amm: {li: 4, le: 2, window: 2}
  - filter: {l_freq: 1, h_freq: 45}
"""


def test_the_elekta_steps(recordings):
    dataset = run_chain(ELEKTA, recordings["elekta"])
    raw = dataset["raw"]
    assert raw.info["sfreq"] == 100
    assert "bad_segment_mag" in raw.annotations.description
    assert any(d.startswith("BAD_muscle") for d in raw.annotations.description)
    assert dataset["events"].shape[1] == 3
    assert len(dataset["epochs"]) > 0
    assert dataset["power"].data.shape[-2] == 4
    assert dataset["itc"] is not None
    assert isinstance(dataset["ica"], list) and len(dataset["ica"]) == 2
    assert all(isinstance(i, mne.preprocessing.ICA) for i in dataset["ica"])


def test_the_ctf_steps(recordings):
    dataset = run_chain(CTF, recordings["ctf"])
    raw = dataset["raw"]
    assert "ref_meg" not in raw.get_channel_types()
    assert "mag" in raw.get_channel_types()


def test_the_eeg_steps(recordings):
    dataset = run_chain(EEG, recordings["eeg"])
    raw = dataset["raw"]
    assert set(raw.get_channel_types()) == {"eeg", "eog"}
    assert raw.get_channel_types(picks=["Fp1"]) == ["eog"]


def test_the_opm_steps(recordings):
    before = mne.io.read_raw_fif(recordings["opm"], verbose=False)
    dataset = run_chain(OPM, recordings["opm"])
    raw = dataset["raw"]
    assert len(raw.info["dig"]) < len(before.info["dig"])
    assert np.isfinite(raw.get_data()).all()


def test_an_unknown_option_is_an_error(recordings):
    with pytest.raises(TypeError):
        run_chain(
            {"preproc": [{"bad_channels": {"picks": "mag", "nope": 1}}]},
            recordings["elekta"],
        )


def test_filtering_a_rectified_signal_warns(recordings, caplog):
    raw = mne.io.read_raw_fif(recordings["elekta"], preload=True, verbose=False)
    raw.crop(0, 5).pick("mag")
    raw._data = np.abs(raw._data)
    with caplog.at_level("WARNING", logger="osl_ephys"):
        run_chain({"preproc": [{"filter": {"l_freq": 1, "h_freq": 40}}]}, raw)
    assert "rectified" in caplog.text
