"""The MNI grid of the forward model: its coordinates and spacing."""

from __future__ import annotations

import numpy as np
import mne
from osl_ephys.session import Session
from osl_ephys.rhino import forward as rhino_forward


def _get_gridstep(coords: np.ndarray) -> int:
    """Get gridstep (i.e. spatial resolution of dipole grid) in mm.

    Parameters
    ----------
    coords : numpy.ndarray
        Coordinates.

    Returns
    -------
    gridstep: int
        Spatial resolution of dipole grid in mm.
    """
    dists = np.linalg.norm(coords - coords[0], axis=-1)
    return int(np.round(dists[dists > 0].min() * 1000))


def _get_source_coords_mni(session: Session, fwd: mne.Forward) -> np.ndarray:
    """Get the coordinates of the in-use dipoles in MNI space.

    Parameters
    ----------
    session : Session
        Files of the session.
    fwd : mne.Forward
        Forward solution.

    Returns
    -------
    src_coords_mni : np.ndarray
        (n_sources, 3) dipole coordinates in MNI space in mm. Ordering
        matches the in-use sources of the forward model (fwd["src"][0]).
    """
    return _get_mni_grid(session, fwd)[fwd["src"][0]["vertno"]]


def _get_mni_grid(session: Session, fwd: mne.Forward) -> np.ndarray:
    """Get the MNI grid of the dipoles in a forward model.

    Parameters
    ----------
    session : Session
        Files of the session.
    fwd : mne.Forward
        Forward solution.

    Returns
    -------
    coords : np.ndarray
        (n_voxels, 3) MNI coordinates (in mm) of each voxel of the grid. The
        dipoles of the forward model are at coords[fwd["src"][0]["vertno"]].
    """
    coords = rhino_forward._mni_grid_coords(session.mni_grid_file)
    if len(coords) != len(fwd["src"][0]["rr"]):
        raise ValueError(
            f"{session.mni_grid_file} does not match {session.fwd_model_file}. "
            "Rerun rhino.forward_model and source_recon.lcmv_beamformer."
        )
    return coords
