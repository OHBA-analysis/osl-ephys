"""Tests for running batch preprocessing - doesn't check that it runs properly,
just that it runs...."""

import unittest
import tempfile
import os

import mne


class TestPreprocessingChain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ..utils import simulate_raw_from_template

        cls.flat_channels = [10]
        cls.bad_channels = [5, 200]
        cls.bad_segments = [(600, 750)]

        cls.raw = simulate_raw_from_template(
            5000,
            flat_channels=cls.flat_channels,
            bad_channels=cls.bad_channels,
            bad_segments=cls.bad_segments,
        )

        cls.fpath = tempfile.NamedTemporaryFile().name + "raw.fif"
        cls.raw.save(cls.fpath)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.fpath)

    def test_simple_chain(self):
        from ..pipeline import run_chain

        cfg = """
        meta:
          event_codes:
        preproc:
          - filter:         {l_freq: 1, h_freq: 30}
          - notch_filter:   {freqs: 50}
          - bad_channels:   {picks: 'grad'}
          - bad_segments:   {segment_len: 800, picks: 'grad'}
        """

        with tempfile.TemporaryDirectory() as outdir:
            dataset = run_chain(
                cfg, {"id": "sub-01", "raw_file": self.fpath}, outdir=outdir
            )

        # Just testing that things run not that the outputs are sensible...
        assert isinstance(dataset["raw"], mne.io.fiff.raw.Raw)


class TestVersions(unittest.TestCase):
    def test_simple_chain(self):
        from ..pipeline.config import check_versions, load_config

        cfg = """
        meta:
          event_codes:
          version_assert:
          version_warn:
        preproc:
          - filter:         {l_freq: 1, h_freq: 30}
          - notch_filter:   {freqs: 50}
          - bad_channels:   {picks: 'grad'}
          - bad_segments:   {segment_len: 800, picks: 'grad'}
        """
        config = load_config(cfg)

        config["meta"]["version_assert"] = ["numpy>1.0", "scipy>1.0"]
        config["meta"]["version_warn"] = ["mne>1.0"]

        check_versions(config)


class TestPreprocessingBatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ..utils import simulate_raw_from_template

        cls.infiles = []

        # First file normal
        cls.raw = simulate_raw_from_template(5000)
        cls.fpath = tempfile.NamedTemporaryFile().name + "raw.fif"
        cls.raw.save(cls.fpath)
        cls.infiles.append(cls.fpath)

        # Second file doesn't exist
        cls.fpath = tempfile.NamedTemporaryFile().name + "raw.fif"
        cls.infiles.append(cls.fpath)

        # Third file normal
        cls.raw = simulate_raw_from_template(5000)
        cls.fpath = tempfile.NamedTemporaryFile().name + "raw.fif"
        cls.raw.save(cls.fpath)
        cls.infiles.append(cls.fpath)

    @classmethod
    def tearDownClass(cls):
        for fpath in cls.infiles:
            if os.path.exists(fpath):
                os.remove(fpath)

    def test_simple_batch(self):
        from ..pipeline import run_batch

        cfg = """
        meta:
          event_codes:
        preproc:
          - bad_channels:   {picks: 'grad'}
          - bad_segments:   {segment_len: 800, picks: 'grad'}
        """

        # The second file does not exist
        with tempfile.TemporaryDirectory() as outdir:
            results = run_batch(cfg, self.infiles, outdir, qc_dir=False)
        goods = [status != "failed" for _, status, _, _ in results]
        assert goods == [True, False, True]

    def test_batch_mixes_existing_and_new_outputs(self):
        """Skipped and processed files must have the same return type."""
        from ..pipeline import run_batch

        cfg = """
        meta:
          event_codes:
        preproc:
          - bad_channels:   {picks: 'grad'}
          - bad_segments:   {segment_len: 800, picks: 'grad'}
        """

        with tempfile.TemporaryDirectory() as outdir:
            first = run_batch(cfg, [self.infiles[0]], outdir, qc_dir=False)
            assert [s for _, s, _, _ in first] != ["failed"]

            # The first output is skipped, while the second is newly made.
            mixed = run_batch(
                cfg, [self.infiles[0], self.infiles[2]], outdir, qc_dir=False
            )
            assert [s != "failed" for _, s, _, _ in mixed] == [True, True]
