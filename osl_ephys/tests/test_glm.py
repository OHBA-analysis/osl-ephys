"""Tests for glm_spectrum and glm_epochs"""

import unittest
import tempfile
import os

import mne
import numpy as np


class TestGLMSpectrum(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from ..utils import simulate_raw_from_template

        cls.flat_channels = None
        cls.bad_channels = None
        cls.bad_segments = None

        cls.raw = simulate_raw_from_template(
            500,
            flat_channels=cls.flat_channels,
            bad_channels=cls.bad_channels,
            bad_segments=cls.bad_segments,
        )

        cls.fpath = tempfile.NamedTemporaryFile().name + "raw.fif"
        cls.raw.save(cls.fpath)

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.fpath)

    def test_glm_spectrum(self):
        from ..glm import glm_spectrum

        spec = glm_spectrum(self.raw)
