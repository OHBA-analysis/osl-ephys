"""Coregistration maths on known inputs: the fiducial transform."""

import numpy as np
import pytest

coreg = pytest.importorskip("osl_ephys.rhino.coregistration")  # needs fslpy


def _rotation(rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _apply(xform, points):
    return xform[:3, :3] @ points + xform[:3, 3:]


def test_rigid_transform_recovers_rotation_and_translation():
    rng = np.random.default_rng(0)
    A = rng.normal(scale=50, size=(3, 3))
    R = _rotation(rng)
    t = rng.normal(scale=20, size=(3, 1))
    B = R @ A + t
    xform, scaling = coreg._rigid_transform_3D(B, A)
    assert np.allclose(scaling, np.eye(4))
    assert np.allclose(_apply(xform, A), B)


def test_scaled_transform_maps_the_fiducials_onto_the_head_fiducials():
    # Template (MNI) fiducials in mm, whose centroid is far from the origin:
    # the scaling (applied first, about the origin) must not shift them
    rng = np.random.default_rng(1)
    A = np.array([[1, 85, -41], [83, -20, -65], [-83, -20, -65]], dtype=float).T
    R = _rotation(rng)
    t = rng.normal(scale=20, size=(3, 1))
    s = 0.85
    B = s * R @ A + t
    xform, scaling = coreg._rigid_transform_3D(B, A, compute_scaling=True)
    assert np.allclose(np.diag(scaling)[:3], s)
    assert np.allclose(_apply(xform @ scaling, A), B, atol=1e-6)
