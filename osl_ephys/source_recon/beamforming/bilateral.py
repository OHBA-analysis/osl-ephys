"""Bilateral pairs of dipoles, for the bilateral beamformer and its plot."""

from __future__ import annotations

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import mne
from osl_ephys.session import Session
from osl_ephys.source_recon.beamforming.grid import (
    _get_gridstep,
    _get_source_coords_mni,
)


def plot_bilateral_pairs(
    session: Session,
    bilateral_tol: float | None = None,
    bilateral_tol_midline: float | None = None,
    filename: str | None = None,
    show: bool = False,
) -> None:
    """Plot the dipole pairs used by the bilateral beamformer.

    The pairing is recomputed from the forward model, so the tolerances passed
    here must match those passed to lcmv_beamformer.

    Parameters
    ----------
    session : Session
        Files of the session.
    bilateral_tol : float, optional
        Tolerance (in mm) for pairing dipoles mirrored across the midline,
        see :func:`lcmv_beamformer`. If None, half the gridstep of the dipole
        grid is used.
    bilateral_tol_midline : float, optional
        Dipoles closer (in mm) than this to the midline are not paired.
        If None, bilateral_tol is used.
    filename : str, optional
        Output filename. If None, {session.src_dir}/bilateral_dipoles.png is used.
    show : bool, optional
        Should we show the plot?
    """
    if filename is None:
        filename = f"{session.src_dir}/bilateral_dipoles.png"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fwd = mne.read_forward_solution(session.fwd_model_file, verbose=False)
    src_coords_mni, multi_dipoles, single_dipoles, midline_points = (
        _find_bilateral_pairs(session, fwd, bilateral_tol, bilateral_tol_midline)
    )
    pairs = np.array(multi_dipoles, dtype=int).reshape(-1, 2)

    views = [("Axial", 0, 1), ("Coronal", 0, 2), ("Sagittal", 1, 2)]
    labels = ["x (mm)", "y (mm)", "z (mm)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (view, i, j) in zip(axes, views):
        ax.plot(
            src_coords_mni[pairs, i].T,
            src_coords_mni[pairs, j].T,
            c="tab:red",
            lw=0.4,
            zorder=1,
        )
        ax.scatter(
            src_coords_mni[midline_points, i],
            src_coords_mni[midline_points, j],
            s=14,
            c="tab:blue",
            zorder=2,
        )
        ax.scatter(
            src_coords_mni[pairs.ravel(), i],
            src_coords_mni[pairs.ravel(), j],
            s=3,
            c="tab:red",
            zorder=3,
        )
        ax.scatter(
            src_coords_mni[single_dipoles, i],
            src_coords_mni[single_dipoles, j],
            s=26,
            facecolors="none",
            edgecolors="grey",
            linewidths=0.7,
            zorder=4,
        )
        ax.set_title(view)
        ax.set_xlabel(labels[i])
        ax.set_ylabel(labels[j])
        ax.set_aspect("equal")

    # Dummy handles for the legend
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=5, mfc="none", mec="grey"),
        plt.Line2D([], [], ls="", marker="o", ms=4, c="tab:blue"),
        plt.Line2D([], [], marker="o", ms=2, c="tab:red"),
    ]
    fig.legend(
        handles,
        ["Single", "Midline", "Bilateral pair"],
        loc="lower center",
        ncol=3,
    )
    fig.suptitle(
        f"{len(pairs)} bilateral pairs, {len(single_dipoles)} single dipoles, "
        f"{len(midline_points)} midline dipoles"
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    print(f"Saving {filename}")
    fig.savefig(filename, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _find_bilateral_pairs(
    session: Session,
    fwd: mne.Forward,
    bilateral_tol: float | None = None,
    bilateral_tol_midline: float | None = None,
) -> tuple[np.ndarray, list[list[int]], np.ndarray, np.ndarray]:
    """Find pairs of bilaterally symmetric dipoles of a forward model.

    The dipoles are mirrored across the plane the MNI grid is symmetric about
    (x = 0, to within half a voxel), so the mirror image of a dipole is on the
    grid.

    Parameters
    ----------
    session : Session
        Files of the session.
    fwd : mne.Forward
        Forward solution.
    bilateral_tol : float, optional
        Tolerance (in mm) for pairing dipoles mirrored across the midline:
        the mean absolute difference between the x, y and z coordinates of a
        dipole and the mirror image of its partner. If None, half the gridstep
        of the dipole grid is used.
    bilateral_tol_midline : float, optional
        Dipoles closer (in mm) than this to the midline are not paired.
        If None, bilateral_tol is used.

    Returns
    -------
    src_coords_mni : np.ndarray
        (n_sources, 3) MNI coordinates (in mm) of the in-use dipoles of the
        forward model.
    multi_dipoles : list of list of int
        Each entry contains the indices (with respect to src_coords_mni) of
        the two dipoles that make up a bilateral pair.
    single_dipoles : np.ndarray
        Indices (with respect to src_coords_mni) of dipoles that could not
        be paired.
    midline_points : np.ndarray
        Indices (with respect to src_coords_mni) of dipoles close to the
        midline.
    """
    src_coords_mni = _get_source_coords_mni(session, fwd)
    img = nib.load(session.mni_grid_file)
    midline_x = nib.affines.apply_affine(
        img.header.get_sform(), [(img.shape[0] - 1) / 2, 0, 0]
    )[0]

    if bilateral_tol is None:
        gridstep = _get_gridstep(src_coords_mni / 1000)
        bilateral_tol = gridstep / 2
        print(f"Setting bilateral_tol = {bilateral_tol} mm")

    if bilateral_tol <= 0:
        raise ValueError("bilateral_tol must be a positive scalar.")

    if bilateral_tol_midline is None:
        bilateral_tol_midline = bilateral_tol
        print(f"Setting bilateral_tol_midline = {bilateral_tol_midline} mm")

    # Dipoles close to the midline are not paired
    coords = src_coords_mni - [midline_x, 0, 0]
    midline_points = np.where(np.abs(coords[:, 0]) < bilateral_tol_midline)[0]
    non_midline_points = np.where(np.abs(coords[:, 0]) >= bilateral_tol_midline)[0]
    coords = coords[non_midline_points]

    # Mirror the right-hemisphere dipoles onto the left hemisphere
    left_points = np.where(coords[:, 0] < 0)[0]
    right_points = np.where(coords[:, 0] >= 0)[0]
    left_coords = coords[left_points]
    right_coords = coords[right_points].copy()
    right_coords[:, 0] = -right_coords[:, 0]

    # Mean absolute difference between all cross-hemisphere dipole pairs
    dist_matrix = np.abs(left_coords[:, np.newaxis] - right_coords[np.newaxis]).mean(
        axis=-1
    )

    # Greedily pair the closest left/right dipoles
    pairs = []
    while dist_matrix.size > 0:
        min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        if dist_matrix[min_idx] > bilateral_tol:
            break
        pairs.append(min_idx)
        dist_matrix[min_idx[0], :] = np.inf
        dist_matrix[:, min_idx[1]] = np.inf

    # Unpaired dipoles, in src_coords_mni indexing
    used_left = np.zeros(len(left_points), dtype=bool)
    used_right = np.zeros(len(right_points), dtype=bool)
    for ll, rr in pairs:
        used_left[ll] = True
        used_right[rr] = True
    single_dipoles = np.concatenate(
        [
            non_midline_points[left_points[~used_left]],
            non_midline_points[right_points[~used_right]],
        ]
    ).astype(int)

    # Pairs in src_coords_mni indexing
    multi_dipoles = [
        [
            int(non_midline_points[left_points[ll]]),
            int(non_midline_points[right_points[rr]]),
        ]
        for ll, rr in pairs
    ]

    n_used = 2 * len(multi_dipoles) + len(single_dipoles) + len(midline_points)
    print(
        f"Found {len(multi_dipoles)} bilateral pairs, "
        f"{len(single_dipoles)} single dipoles and "
        f"{len(midline_points)} midline dipoles "
        f"({n_used} of {src_coords_mni.shape[0]} dipoles)"
    )

    return src_coords_mni, multi_dipoles, single_dipoles, midline_points
