"""run_chain and run_batch on the test recordings."""

import csv

import mne
import pytest

from osl_ephys import files
from osl_ephys.pipeline import run_batch, run_chain

CONFIG = """
preproc:
  - filter: {l_freq: 1, h_freq: 40}
  - resample: {sfreq: 100}
  - bad_segments: {segment_len: 100, picks: mag}
"""


@pytest.fixture(scope="module")
def sessions():
    return [
        {"id": "elekta", "subject": "sub-01", "raw_file": files.test_data.elekta},
        {"id": "ctf", "subject": "sub-02", "raw_file": files.test_data.ctf},
        {"id": "missing", "subject": "sub-03", "raw_file": "/nowhere/missing-raw.fif"},
    ]


def test_run_chain_in_memory(sessions):
    dataset = run_chain(CONFIG, sessions[0])
    assert isinstance(dataset["raw"], mne.io.BaseRaw)
    assert dataset["raw"].info["sfreq"] == 100
    assert "bad_segment" in " ".join(dataset["raw"].annotations.description)
    assert dataset["session"]["id"] == "elekta"


def test_run_batch_writes_outputs_and_isolates_failures(sessions, tmp_path):
    results = run_batch(CONFIG, sessions, tmp_path, n_workers=2, qc_dir=False)

    status = {id_: s for id_, s, _, _ in results}
    assert status == {"elekta": "ok", "ctf": "ok", "missing": "failed"}

    for id_ in ("elekta", "ctf"):
        out = tmp_path / "preproc" / f"{id_}_preproc-raw.fif"
        assert out.exists(), out
        assert mne.io.read_raw_fif(out, verbose=False).info["sfreq"] == 100
        assert (tmp_path / "logs" / f"{id_}.log").exists()
    assert (tmp_path / "logs" / "missing.log.err").exists()
    with open(tmp_path / "logs" / "summary.csv") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    assert rows["missing"]["status"] == "failed"

    # A second run skips the sessions whose output exists
    results = run_batch(CONFIG, sessions[:2], tmp_path, n_workers=1, qc_dir=False)
    assert all(s == "ok" for _, s, _, _ in results)
    assert "Skipping preprocessing" in (tmp_path / "logs" / "elekta.log").read_text()


def test_run_batch_with_the_report(sessions, tmp_path):
    results = run_batch(CONFIG, sessions[:1], tmp_path, n_workers=1)
    assert results[0][1] == "ok"
    assert (tmp_path / "qc" / "report.html").exists()
    assert (tmp_path / "qc" / "preproc" / "elekta" / "summary.json").exists()
    assert (tmp_path / "qc" / "preproc" / "elekta" / "psd.webp").exists()
