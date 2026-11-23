"""Sign flipping recovers known flips on synthetic parcel data."""

import json

import mne
import numpy as np
import pytest

from osl_ephys.group import sign_flipping, steps
from osl_ephys.group.steps import _parc_files
from osl_ephys.utils import processing

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


@pytest.fixture(scope="module")
def covs(data):
    template, flipped, _ = data
    return (
        sign_flipping.compute_covariance(flipped, N_EMBEDDINGS),
        sign_flipping.compute_covariance(template, N_EMBEDDINGS),
    )


def _agrees(found, truth):
    """The flips up to a global sign."""
    found = np.asarray(found)
    return np.array_equal(found, truth) or np.array_equal(-found, truth)


def _corr(cov, template_cov):
    return sign_flipping.covariance_matrix_correlation(
        cov, template_cov, diag_offset=N_EMBEDDINGS
    )


def test_the_methods_are_those_of_osl_signflip():
    assert set(sign_flipping.METHODS) == {
        "sdp",
        "spectral",
        "belief_propagation",
        "random",
    }
    assert sign_flipping.DEFAULT_METHOD == "sdp"


@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("sdp", {}),
        ("spectral", {}),
        ("belief_propagation", {}),
        # The default max_flips (20) exceeds the number of synthetic parcels
        ("random", {"max_flips": N_PARCELS // 2}),
    ],
)
def test_every_method_recovers_the_flips(data, covs, method, kwargs):
    _, _, flips = data
    cov, template_cov = covs
    found, corr = sign_flipping.find_flips(
        cov, template_cov, N_EMBEDDINGS, method, **kwargs
    )
    assert _agrees(found, flips)
    assert corr > _corr(cov, template_cov) and corr > 0.99


def test_the_correlation_is_that_of_the_applied_flips(covs):
    # The returned correlation is the last entry of the method's metric history,
    # which must be that of the applied flips even when the history is padded
    # (belief propagation)
    cov, template_cov = covs
    found, corr = sign_flipping.find_flips(
        cov, template_cov, N_EMBEDDINGS, "belief_propagation"
    )
    flipped_cov = sign_flipping.apply_flips_to_covariance(cov, found, N_EMBEDDINGS)
    assert corr == pytest.approx(_corr(flipped_cov, template_cov))


def test_sdp_is_the_default(covs):
    cov, template_cov = covs
    default, _ = sign_flipping.find_flips(cov, template_cov, N_EMBEDDINGS)
    sdp, _ = sign_flipping.find_flips(cov, template_cov, N_EMBEDDINGS, "sdp")
    np.testing.assert_array_equal(default, sdp)


def test_the_options_reach_the_method(covs):
    cov, template_cov = covs
    with pytest.raises(ValueError, match="max_flips"):
        # More flips per iteration than there are channels
        sign_flipping.find_flips(
            cov, template_cov, N_EMBEDDINGS, "random", max_flips=N_PARCELS + 1
        )


def test_get_method_hyperparameters_lists_them():
    assert sign_flipping.get_method_hyperparameters("random") == {
        "n_restarts": 3,
        "n_iter": 2500,
        "max_flips": 20,
        "use_tqdm": True,
        "random_state": 0,
    }
    assert "sdp_n_restarts" in sign_flipping.get_method_hyperparameters("sdp")


def test_invalid_methods_and_options_are_errors(covs):
    cov, template_cov = covs
    for method in ["coordinate_ascent", "SDP"]:
        with pytest.raises(ValueError, match="Invalid sign flipping method"):
            sign_flipping.find_flips(cov, template_cov, N_EMBEDDINGS, method)
    with pytest.raises(ValueError, match="'n_iter'.*Valid options are"):
        sign_flipping.find_flips(cov, template_cov, N_EMBEDDINGS, "sdp", n_iter=10)


def test_the_covariance_matches_osl_ephys_processing(data):
    template, _, _ = data
    embedded = processing.standardize(processing.time_embed(template, N_EMBEDDINGS))
    np.testing.assert_allclose(
        sign_flipping.compute_covariance(template, N_EMBEDDINGS),
        np.cov(embedded.T),
        rtol=1e-12,
        atol=1e-14,
    )


def test_sign_flip_applies_the_flips(data):
    template, flipped, flips = data
    template_cov = sign_flipping.compute_covariance(template, N_EMBEDDINGS)
    out, found, corr = sign_flipping.sign_flip(flipped.T, template_cov, N_EMBEDDINGS)
    assert out.shape == flipped.T.shape
    assert np.allclose(out, flipped.T * np.asarray(found)[:, None])
    assert _agrees(found, flips)
    assert corr > 0.99


def test_the_parcellated_files_are_found_by_estimator(tmp_path):
    for name in ["s1/s1_lcmv-parc-raw.fif", "s1/s1_sflip-lcmv-parc-raw.fif"]:
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).touch()
    (tmp_path / "s2").mkdir()
    estimator, files = _parc_files(tmp_path, ["s1", "s2"])
    assert estimator == "lcmv"
    assert files == {"s1": tmp_path / "s1" / "s1_lcmv-parc-raw.fif"}

    (tmp_path / "s2" / "s2_eloreta-parc-raw.fif").touch()
    with pytest.raises(ValueError, match="eloreta, lcmv"):
        _parc_files(tmp_path, ["s1", "s2"])
    estimator, files = _parc_files(tmp_path, ["s1", "s2"], "eLORETA")
    assert estimator == "eloreta"
    assert files == {"s2": tmp_path / "s2" / "s2_eloreta-parc-raw.fif"}


def _write_parc(path, x):
    """Save (n_samples, n_parcels) data as a parcellated fif file."""
    info = mne.create_info([f"parcel_{i}" for i in range(x.shape[1])], 250, "misc")
    path.parent.mkdir(parents=True, exist_ok=True)
    mne.io.RawArray(x.T, info, verbose=False).save(path, verbose=False)


def _read_parc(path):
    return mne.io.read_raw_fif(path, verbose=False).get_data(picks="misc").T


@pytest.fixture
def parc_dir(tmp_path, data):
    """Three sessions: the template, a flipped copy, and a noisier flipped copy."""
    template, flipped, flips = data
    rng = np.random.default_rng(2)
    noisy = flipped + 0.05 * rng.normal(size=flipped.shape)
    for id_, x in [("a", template), ("b", flipped), ("c", noisy)]:
        _write_parc(tmp_path / id_ / f"{id_}_lcmv-parc-raw.fif", x)
    return tmp_path, flips


def test_the_group_step_flips_every_session(parc_dir):
    outdir, flips = parc_dir
    dataset = {
        "sessions": [{"id": i} for i in "abc"],
        "outdir": outdir,
        "qc_dir": outdir / "qc",
    }
    userargs = {"n_embeddings": N_EMBEDDINGS, "method": "spectral", "template": "a"}
    dataset = steps.sign_flip(dataset, userargs)
    assert dataset["template"] == "a"

    out = {i: _read_parc(outdir / i / f"{i}_sflip-lcmv-parc-raw.fif") for i in "abc"}
    inp = {i: _read_parc(outdir / i / f"{i}_lcmv-parc-raw.fif") for i in "abc"}
    np.testing.assert_array_equal(out["a"], inp["a"])  # the template is not flipped
    for i in "bc":
        found = np.sign(np.sum(out[i] * inp[i], axis=0))
        assert _agrees(found, flips)

    summary = json.loads((outdir / "qc" / "sign_flip" / "summary.json").read_text())
    assert summary["template"] == "a" and summary["method"] == "spectral"
    assert summary["b"]["after"] > summary["b"]["before"]


def test_the_group_step_picks_a_template(parc_dir):
    outdir, _ = parc_dir
    dataset = {"sessions": [{"id": i} for i in "abc"], "outdir": outdir}
    dataset = steps.sign_flip(dataset, {"n_embeddings": N_EMBEDDINGS})
    assert dataset["template"] in "abc"


def test_the_steps_reject_unknown_options_before_reading(tmp_path):
    dataset = {"sessions": [{"id": "a"}], "outdir": tmp_path}
    with pytest.raises(ValueError, match="'n_random_starts'.*Valid options are"):
        steps.sign_flip(dataset, {"n_random_starts": 3})
    with pytest.raises(ValueError, match="Invalid sign flipping method"):
        steps.sign_flip(dataset, {"method": "coordinate_ascent"})


def test_bad_segments_are_left_out_of_the_covariance(tmp_path, data):
    template, flipped, flips = data
    # A bad segment of large noise in the middle of the recording
    corrupted = flipped.copy()
    corrupted[1000:2000] = 100 * np.random.default_rng(3).normal(size=(1000, N_PARCELS))
    info = mne.create_info([f"parcel_{i}" for i in range(N_PARCELS)], 250, "misc")
    raw = mne.io.RawArray(corrupted.T, info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=1000 / 250, duration=1000 / 250, description="bad_segment"
        )
    )
    parc = tmp_path / "b" / "b_lcmv-parc-raw.fif"
    parc.parent.mkdir()
    raw.save(parc, verbose=False)

    # The data as saved (fif files store single precision), without the bad segment
    stored = _read_parc(parc)
    good = np.concatenate([stored[:1000], stored[2000:]])
    np.testing.assert_allclose(
        steps._cov(parc, N_EMBEDDINGS, True),
        sign_flipping.compute_covariance(good, N_EMBEDDINGS),
    )

    # The flips are still applied to the whole recording, bad segment included
    template_cov = sign_flipping.compute_covariance(template, N_EMBEDDINGS)
    out = parc.with_name("b_sflip-lcmv-parc-raw.fif")
    steps._flip_file(parc, template_cov, out, N_EMBEDDINGS, True, "sdp", {})
    saved = _read_parc(out)
    found = np.sign(np.sum(saved * stored, axis=0))
    assert saved.shape == stored.shape
    assert _agrees(found, flips)
    np.testing.assert_array_equal(saved, stored * found)
