"""Regression tests for the reviewed bad-segment time origin."""

import mne
import numpy as np
import pytest

from osl_ephys.preprocessing.manual_ica.apply import apply_one


@pytest.mark.parametrize("first_samp", [0, 37])
def test_apply_one_places_bad_segments_relative_to_cropped_raw(
    tmp_path, monkeypatch, first_samp,
):
    subject = "11"
    raw_dir = tmp_path / "raw" / subject
    review_dir = tmp_path / "review" / subject
    raw_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)

    raw = mne.io.RawArray(
        np.zeros((1, 200)), mne.create_info(["Pz"], 10.0, "eeg"),
        first_samp=first_samp, verbose="ERROR",
    )
    raw.save(raw_dir / "11_preproc-raw.fif", verbose="ERROR")
    (raw_dir / "11_ica.fif").touch()
    (review_dir / "label.txt").write_text("IC000: good\n")
    (review_dir / "bads.txt").write_text("5 7\n18 20\n")

    class IdentityICA:
        n_components_ = 1
        exclude = []

        def apply(self, raw, verbose=False):
            return raw

    monkeypatch.setattr(mne.preprocessing, "read_ica", lambda *a, **k: IdentityICA())
    status = apply_one(tmp_path / "review", tmp_path / "raw", subject)
    assert status.startswith("ok ---")

    result = mne.io.read_raw_fif(
        raw_dir / "11_after_ica-raw.fif", preload=True, verbose="ERROR",
    )
    manual = [a for a in result.annotations if a["description"] == "BAD_manual"]
    assert len(manual) == 2
    np.testing.assert_allclose(
        [a["onset"] - result.first_time for a in manual], [5.0, 18.0],
    )
    np.testing.assert_allclose(
        [a["duration"] for a in manual], [2.0, 2.0], atol=2e-6,
    )
    masked = result.get_data(picks="Pz", reject_by_annotation="NaN")[0]
    assert np.isnan(masked[50:70]).all()
    assert np.isnan(masked[180:]).all()
