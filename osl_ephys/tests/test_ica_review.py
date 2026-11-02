"""Applying hand-labelled ICA components to a run's preprocessed data."""

import mne
import numpy as np
import pytest

from osl_ephys import files
from osl_ephys.preproc.ica import apply_review
from osl_ephys.preproc.ica.files import ica_files, sessions_with_ica


@pytest.fixture
def run(tmp_path):
    """A run with one session, its preprocessed data and a small ICA."""
    raw = mne.io.read_raw_fif(files.test_data.elekta, preload=True, verbose=False)
    raw.pick("mag").filter(1, 40, verbose=False).resample(100, verbose=False)
    ica = mne.preprocessing.ICA(n_components=5, random_state=0, max_iter=200)
    ica.fit(raw, verbose=False)
    preproc = tmp_path / "preproc"
    preproc.mkdir()
    raw.save(preproc / "elekta_preproc-raw.fif", verbose=False)
    ica.save(preproc / "elekta_ica.fif", verbose=False)
    return tmp_path, raw, ica


def test_the_gui_route_applies_the_saved_exclusions(run):
    outdir, raw, ica = run
    ica.exclude = [0, 1]
    ica.save(ica_files(outdir, "elekta")[0], overwrite=True, verbose=False)
    assert sessions_with_ica(outdir) == ["elekta"]
    message = apply_review(outdir, "elekta")
    assert message.startswith("ok")
    cleaned = mne.io.read_raw_fif(outdir / "preproc" / "elekta_preproc-raw.fif")
    assert not np.allclose(cleaned.get_data(), raw.get_data(), rtol=1e-3, atol=0)


def test_the_browser_route_reads_the_label_file(run):
    outdir, raw, ica = run
    review = outdir / "ica" / "elekta"
    review.mkdir(parents=True)
    labels = ["bad" if i < 2 else "good" for i in range(ica.n_components_)]
    review.joinpath("label.txt").write_text(
        "\n".join(f"IC{i:03d}: {label}" for i, label in enumerate(labels))
    )
    review.joinpath("bads.txt").write_text("1.0 2.0\n")
    message = apply_review(outdir, "elekta")
    assert message.startswith("ok")
    cleaned = mne.io.read_raw_fif(outdir / "preproc" / "elekta_preproc-raw.fif")
    assert not np.allclose(cleaned.get_data(), raw.get_data(), rtol=1e-3, atol=0)
    assert "BAD_manual" in cleaned.annotations.description
