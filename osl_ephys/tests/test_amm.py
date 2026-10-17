"""AMM interference rejection on the OPM test recording."""

import mne
import numpy as np

from osl_ephys import files
from osl_ephys.preproc.amm import apply_amm


def test_apply_amm_reduces_the_variance():
    raw = mne.io.read_raw_fif(files.test_data.opm, preload=True, verbose=False)
    raw.crop(0, 5)
    cleaned, info = apply_amm(raw, li=4, le=2, window=2.0)
    picks = mne.pick_types(raw.info, meg=True, exclude="bads")
    before = raw.get_data(picks).var(axis=1).sum()
    after = cleaned.get_data(picks).var(axis=1).sum()
    assert cleaned.get_data().shape == raw.get_data().shape
    assert np.isfinite(after) and after < before
    assert set(info) >= {"center", "radii"}
