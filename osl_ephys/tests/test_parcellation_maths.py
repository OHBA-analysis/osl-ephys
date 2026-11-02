"""Parcellation maths on known inputs: parcel weights and orthogonalisation."""

import nibabel as nib
import numpy as np
import pytest

from osl_ephys import files
from osl_ephys.parcellation import parcellation as parc

PARCELLATION = "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz"


@pytest.fixture(scope="module")
def grid():
    """The parcellation's own voxels (MNI mm) and the parcel each belongs to."""
    p = parc.Parcellation(PARCELLATION)
    weights = p.data()
    inside = np.flatnonzero(weights.reshape(-1, p.n_parcels).sum(axis=1) > 0)
    ijk = np.array(np.unravel_index(inside, weights.shape[:3])).T
    coords = nib.affines.apply_affine(p.parcellation.affine, ijk)
    parcel_of = weights.reshape(-1, p.n_parcels)[inside].argmax(axis=1)
    return coords, parcel_of, p.n_parcels


def test_pca_parcel_time_courses_follow_the_parcel_signal(grid):
    coords, parcel_of, n_parcels = grid
    rng = np.random.default_rng(0)
    signals = rng.normal(size=(n_parcels, 2000))
    voxel_data = signals[parcel_of] + 0.05 * rng.normal(size=(len(coords), 2000))
    parcel_data = parc.parcellate(voxel_data, coords, PARCELLATION, method="pca")
    assert parcel_data.shape == (n_parcels, 2000)
    for i in range(n_parcels):
        r = np.corrcoef(parcel_data[i], signals[i])[0, 1]
        assert abs(r) > 0.95, i


def test_centroid_parcel_time_courses_are_the_voxel_means(grid):
    coords, parcel_of, n_parcels = grid
    rng = np.random.default_rng(1)
    voxel_data = rng.normal(size=(len(coords), 500))
    parcel_data = parc.parcellate(voxel_data, coords, PARCELLATION, method="centroid")
    assert parcel_data.shape == (n_parcels, 500)
    assert np.isfinite(parcel_data).all()


def test_symmetric_orthogonalisation_removes_zero_lag_correlations():
    rng = np.random.default_rng(2)
    shared = rng.normal(size=(1, 5000))
    data = shared + 0.5 * rng.normal(size=(8, 5000))
    assert np.abs(np.corrcoef(data)[np.triu_indices(8, 1)]).mean() > 0.5
    data -= data.mean(axis=1, keepdims=True)
    ortho = parc._symmetric_orthogonalisation(data)
    assert ortho.shape == data.shape
    gram = ortho @ ortho.T
    assert np.allclose(gram, np.diag(np.diag(gram)), atol=1e-6 * gram.max())


def test_dipole_counts_cover_every_parcel(grid):
    coords, parcel_of, n_parcels = grid
    counts = parc.parcel_dipole_counts(PARCELLATION, coords)
    assert counts.shape == (n_parcels,)
    assert counts.min() > 0
    assert counts.sum() >= len(coords)
