"""The QC plots and the report on the test recordings."""

import json

import mne
import pytest

from osl_ephys import files
from osl_ephys.report import generate_report
from osl_ephys.preproc.qc import save_qc_plots, gen_report_from_fif


@pytest.fixture(scope="module")
def raw():
    raw = mne.io.read_raw_fif(files.test_data.elekta, preload=True, verbose=False)
    raw.filter(1, 40, verbose=False).resample(100, verbose=False)
    raw.info["bads"] = [raw.ch_names[0]]
    raw.set_annotations(mne.Annotations([1.0], [0.5], ["bad_segment_mag"]))
    return raw


def test_save_qc_plots_and_report(raw, tmp_path):
    summary = save_qc_plots(tmp_path, "elekta", raw)
    qc = tmp_path / "preproc" / "elekta"
    assert json.loads((qc / "summary.json").read_text()) == summary
    assert summary["n_bad_channels"] == 1
    assert summary["bad_percent_mag"] > 0
    for name in ["psd", "sum_square", "channel_stds", "sensors", "digitisation"]:
        assert (qc / f"{name}.webp").exists(), name

    path = generate_report(tmp_path, [{"id": "elekta"}])
    report = path.read_text()
    assert "Bad segments (%)" in report and '"elekta"' in report


def test_custom_figures_get_a_tab(raw, tmp_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([0, 1])
    save_qc_plots(tmp_path, "elekta", raw, custom_figures={"line": fig}, plots=["psd"])
    assert (tmp_path / "preproc" / "elekta" / "custom_line.webp").exists()
    report = generate_report(tmp_path, [{"id": "elekta"}]).read_text()
    assert "Custom" in report and "custom_line.webp" in report


def test_epochs_and_manifest_columns(raw, tmp_path):
    events = mne.make_fixed_length_events(raw, duration=2.0)
    events[::2, 2] = 2
    epochs = mne.Epochs(
        raw, events, event_id={"a": 1, "b": 2}, tmin=-0.2, tmax=0.5, preload=True
    )
    summary = save_qc_plots(tmp_path, "elekta", raw, epochs=epochs, plots=["evoked"])
    assert summary["n_epochs"] == len(epochs)
    assert (tmp_path / "preproc" / "elekta" / "evoked.webp").exists()
    report = generate_report(
        tmp_path, [{"id": "elekta", "notes": "moved a lot", "raw_file": "x.fif"}]
    ).read_text()
    assert "Epochs" in report and "notes" in report and "moved a lot" in report
    assert "raw_file" not in report


def test_gen_report_from_fif(raw, tmp_path):
    fif = tmp_path / "elekta_preproc-raw.fif"
    raw.save(fif, verbose=False)
    path = gen_report_from_fif([str(fif)], tmp_path / "qc", plots=["psd"])
    assert path.exists()
    assert (tmp_path / "qc" / "preproc" / "elekta" / "psd.webp").exists()
