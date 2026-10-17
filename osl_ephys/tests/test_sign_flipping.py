"""Sign flipping recovers known flips on synthetic parcel data."""

import numpy as np
import pytest

from osl_ephys.group import sign_flipping

N_PARCELS, N_SAMPLES, N_EMBEDDINGS = 12, 4000, 5


@pytest.fixture(scope="module")
def data():
    """Correlated parcel time courses, and a copy with half the signs flipped."""
    rng = np.random.default_rng(1)
    mixing = rng.normal(size=(N_PARCELS, 4))
    sources = rng.normal(size=(N_SAMPLES, 4))
    sources = np.cumsum(sources, axis=0)
    sources -= sources.mean(axis=0)
    template = sources @ mixing.T + 0.1 * rng.normal(size=(N_SAMPLES, N_PARCELS))
    flips = np.ones(N_PARCELS)
    flips[rng.choice(N_PARCELS, N_PARCELS // 2, replace=False)] = -1
    return template, template * flips, flips


def _agrees(found, truth):
    """The flips up to a global sign."""
    found = np.asarray(found)
    return np.array_equal(found, truth) or np.array_equal(-found, truth)


def test_coordinate_ascent_recovers_the_flips(data):
    template, flipped, flips = data
    template_cov = sign_flipping.calc_cov(template, N_EMBEDDINGS)
    cov = sign_flipping.calc_cov(flipped, N_EMBEDDINGS)
    before = sign_flipping.calc_corr(cov, template_cov, N_EMBEDDINGS)
    found, after = sign_flipping.find_flips(cov, template_cov, N_PARCELS, N_EMBEDDINGS)
    assert _agrees(found, flips)
    assert after > before and after > 0.99


@pytest.mark.parametrize("solver", ["find_flips_spectral", "find_flips_sdp"])
def test_the_solvers_recover_the_flips(data, solver):
    template, flipped, flips = data
    template_cov = sign_flipping.calc_cov(template, N_EMBEDDINGS)
    cov = sign_flipping.calc_cov(flipped, N_EMBEDDINGS)
    found, metrics = getattr(sign_flipping, solver)(cov, template_cov, N_EMBEDDINGS)
    assert _agrees(found, flips)
    assert metrics[1] > metrics[0]


def test_sign_flip_applies_the_flips(data):
    template, flipped, flips = data
    template_cov = sign_flipping.calc_cov(template, N_EMBEDDINGS)
    out, found, corr = sign_flipping.sign_flip(flipped.T, template_cov, N_EMBEDDINGS)
    assert out.shape == flipped.T.shape
    assert np.allclose(out, flipped.T * np.asarray(found)[:, None])
    assert corr > 0.99
