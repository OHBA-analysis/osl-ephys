"""Regression tests for applying manual ICA review decisions safely."""

import sys

import mne
import numpy as np
import pytest

from osl_ephys.preprocessing.manual_ica.apply import apply_one, main
from osl_ephys.preprocessing.manual_ica.io import parse_bads_txt


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


@pytest.mark.parametrize('labels', [
    'IC000: good\n',  # Truncated review: IC001 is missing.
    'IC000: good\nIC000: bad\nIC001: good\n',  # Conflicting labels.
    'IC000: good\nIC001: good\nIC002: good\n',  # Extra good label.
    'IC000: good\nIC001: unlabeled\n',
])
def test_apply_one_rejects_incomplete_or_duplicate_labels(
    tmp_path, monkeypatch, labels,
):
    subject = '11'
    raw_dir = tmp_path / 'raw' / subject
    review_dir = tmp_path / 'review' / subject
    raw_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)
    (raw_dir / '11_preproc-raw.fif').touch()
    (raw_dir / '11_ica.fif').touch()
    (review_dir / 'label.txt').write_text(labels)

    class TwoComponentICA:
        n_components_ = 2

    monkeypatch.setattr(mne.preprocessing, 'read_ica', lambda *a, **k: TwoComponentICA())
    status = apply_one(tmp_path / 'review', tmp_path / 'raw', subject)
    assert status.startswith('skip ---')
    assert not (raw_dir / '11_after_ica-raw.fif').exists()


@pytest.mark.parametrize('line', [
    '5\n', 'five 7\n', '7 5\n', '-1 2\n', 'nan 7\n', '5 inf\n', '5 7 extra\n',
])
def test_parse_bads_rejects_malformed_intervals(tmp_path, line):
    path = tmp_path / 'bads.txt'
    path.write_text('# reviewed bad segments\n' + line)
    with pytest.raises(ValueError, match=r'bads\.txt:2:'):
        parse_bads_txt(path)


def test_parse_bads_allows_comments(tmp_path):
    path = tmp_path / 'bads.txt'
    path.write_text('# reviewed bad segments\n5 7 # blink\n')
    assert parse_bads_txt(path) == [(5.0, 2.0)]


def test_apply_one_does_not_write_when_bad_intervals_are_malformed(
    tmp_path, monkeypatch,
):
    subject = '11'
    raw_dir = tmp_path / 'raw' / subject
    review_dir = tmp_path / 'review' / subject
    raw_dir.mkdir(parents=True)
    review_dir.mkdir(parents=True)
    (raw_dir / '11_preproc-raw.fif').touch()
    (raw_dir / '11_ica.fif').touch()
    (review_dir / 'label.txt').write_text('IC000: good\n')
    (review_dir / 'bads.txt').write_text('7 5\n')

    class OneComponentICA:
        n_components_ = 1

    monkeypatch.setattr(mne.preprocessing, 'read_ica', lambda *a, **k: OneComponentICA())
    with pytest.raises(ValueError, match=r'bads\.txt:1:'):
        apply_one(tmp_path / 'review', tmp_path / 'raw', subject)
    assert not (raw_dir / '11_after_ica-raw.fif').exists()


def test_cli_exits_nonzero_when_requested_subject_is_skipped(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', [
        'osl-ica-apply', str(tmp_path), str(tmp_path), 'missing',
    ])
    assert main() == 1
    assert '1 skipped' in capsys.readouterr().out
