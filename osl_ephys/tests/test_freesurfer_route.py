"""The FreeSurfer route on MNE's fsaverage: coregistration, surface forward
model, minimum norm and surface parcellation, without FreeSurfer itself."""

import os
from pathlib import Path

import mne
import pytest

from osl_ephys import files
from osl_ephys.pipeline import run_batch

FSAVERAGE = (
    Path(
        os.environ.get("SUBJECTS_DIR", Path.home() / "mne_data" / "MNE-fsaverage-data")
    )
    / "fsaverage"
)

CONFIG = """
preproc:
  - resample: {sfreq: 100}
coreg:
  - recon_all: {}
  - coregister: {n_init: 5}
source_recon:
  - forward_model: {spacing: oct5, ico: 3}
  - minimum_norm: {chantypes: [mag, grad], rank: {meg: 60}}
parcellation:
  - parcellate: {source_method: minimum_norm, parcellation_file: aparc, method: mean_flip}
"""


@pytest.mark.skipif(
    not FSAVERAGE.exists(),
    reason="MNE's fsaverage is not downloaded (mne.datasets.fetch_fsaverage)",
)
def test_freesurfer_route_on_fsaverage(tmp_path):
    (tmp_path / "freesurfer").mkdir()
    (tmp_path / "freesurfer" / "fsaverage").symlink_to(FSAVERAGE)
    sessions = [
        {"id": "elekta", "subject": "fsaverage", "raw_file": files.test_data.elekta}
    ]

    results = run_batch(
        CONFIG,
        sessions,
        tmp_path,
        n_workers=1,
        qc_dir=False,
        surface_extraction_method="freesurfer",
    )
    assert results[0][1] == "ok", results

    fs = tmp_path / "elekta" / "freesurfer"
    for name in (
        "coreg-trans.fif",
        "coreg.json",
        "space-src.fif",
        "model-fwd.fif",
        "op-inv.fif",
    ):
        assert (fs / name).exists(), name
    parc = mne.io.read_raw_fif(
        tmp_path / "elekta" / "elekta_parc-raw.fif", verbose=False
    )
    assert len(mne.pick_types(parc.info, misc=True)) == 68
    assert "bankssts-lh" in parc.ch_names
