"""Fiducials and headshape points: extraction from fif, Polhemus and Zebris files, and cleaning them up."""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.io.constants import FIFF
from osl_ephys.session import Session


def extract_fiducials_and_headshape_from_fif(
    session: Session,
    include_eeg_as_headshape: bool = False,
    include_hpi_as_headshape: bool = True,
    include_extra_as_headshape: bool = True,
) -> None:
    """Extract headshape points and fiducials from FIF info.

    Extract (polhemus) fiducials and headshape points from MNE raw.info and
    write them out in the required file format for RHINO (in HEAD space in
    mm).

    Should only be used with MNE-derived .fif files that have the expected
    digitised points held in info['dig'] of fif_file.

    Parameters
    ----------
    session : Session
        Files of the session.
    include_eeg_as_headshape : bool, optional
        Should we include EEG locations as headshape points?
    include_hpi_as_headshape : bool, optional
        Should we include HPI locations as headshape points?
    include_extra_as_headshape : bool, optional
        Should we include the extra locations as headshape points?
        These are the unnamed points digitised over the scalp.
    """
    print()
    print("Extracting fiducials/headshape points from fif info")
    print("---------------------------------------------------")

    # Read info from fif file
    info = mne.io.read_info(session.preproc_file)

    # Lists to hold polhemus data
    headshape = []
    rpa = []
    lpa = []
    nasion = []

    # Get fiducials/headshape points
    for dig in info["dig"]:

        # Check dig is in HEAD/Polhemus space
        if dig["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
            raise ValueError(f"{dig['ident']} is not in HEAD space")

        if dig["kind"] == FIFF.FIFFV_POINT_CARDINAL:
            if dig["ident"] == FIFF.FIFFV_POINT_LPA:
                lpa = dig["r"]
            elif dig["ident"] == FIFF.FIFFV_POINT_RPA:
                rpa = dig["r"]
            elif dig["ident"] == FIFF.FIFFV_POINT_NASION:
                nasion = dig["r"]
            else:
                raise ValueError(f"Unknown fiducial: {dig['ident']}")
        elif dig["kind"] == FIFF.FIFFV_POINT_EXTRA and include_extra_as_headshape:
            headshape.append(dig["r"])
        elif dig["kind"] == FIFF.FIFFV_POINT_EEG and include_eeg_as_headshape:
            headshape.append(dig["r"])
        elif dig["kind"] == FIFF.FIFFV_POINT_HPI and include_hpi_as_headshape:
            headshape.append(dig["r"])

    headshape = np.array(headshape)
    rpa = np.array(rpa)
    lpa = np.array(lpa)
    nasion = np.array(nasion)

    # Check if info is from a CTF scanner
    if info["dev_ctf_t"] is not None:
        print("Detected CTF data")

        nasion = np.copy(nasion)
        lpa = np.copy(lpa)
        rpa = np.copy(rpa)

        nasion[0], nasion[1], nasion[2] = nasion[1], -nasion[0], nasion[2]
        lpa[0], lpa[1], lpa[2] = lpa[1], -lpa[0], lpa[2]
        rpa[0], rpa[1], rpa[2] = rpa[1], -rpa[0], rpa[2]

        # CTF data won't contain headshape points, use a dummy point to avoid errors
        headshape = [0, 0, 0]

    # Save
    os.makedirs(session.coreg_dir, exist_ok=True)
    print(f"Saved: {session.coreg.head_nasion_file}")
    np.savetxt(session.coreg.head_nasion_file, nasion * 1000)
    print(f"Saved: {session.coreg.head_rpa_file}")
    np.savetxt(session.coreg.head_rpa_file, rpa * 1000)
    print(f"Saved: {session.coreg.head_lpa_file}")
    np.savetxt(session.coreg.head_lpa_file, lpa * 1000)
    print(f"Saved: {session.coreg.head_headshape_file}")
    np.savetxt(session.coreg.head_headshape_file, np.array(headshape).T * 1000)

    if info["dev_ctf_t"] is not None:
        print(
            "Dummy headshape points saved, overwrite "
            f"{session.coreg.head_headshape_file} "
            "or set use_headshape=False in coregisteration."
        )

    # Warning if 'trans' in filename we assume -trans was applied using MaxFiltering
    # This may make the coregistration appear incorrect, but this is not an issue.
    if "_trans" in session.preproc_file:
        print(
            "fif filename contains '_trans' which suggests -trans was passed "
            "during MaxFiltering. This means the location of the head in the "
            "coregistration plot may not be correct. Either use the _tsss.fif "
            "file or ignore the centroid of the head in coregistration plot."
        )


def extract_fiducials_and_headshape_from_pos(session: Session) -> None:
    """Saves fiducials/headshape from a pos file.

    Two layouts are supported. Either the headshape points are the first rows
    (the header gives how many) and the fiducials are labelled ``nasion``,
    ``left`` and ``right``, or the headshape points are labelled ``EXTRA`` and
    the fiducials ``nasion``, ``lpa`` and ``rpa``. Labels are case-insensitive.

    Parameters
    ----------
    session : Session
        Files of the session.
    """
    if session.pos_file is None:
        raise ValueError("pos_file must have been passed to Session")

    print(f"Saving fiducials/headshape points from {session.pos_file}")

    # These values are in cm in HEAD space
    num_headshape_pnts = int(
        pd.read_csv(session.pos_file, header=None).to_numpy()[0, 0]
    )
    data = pd.read_csv(session.pos_file, header=None, skiprows=[0], sep=r"\s+")

    # RHINO is going to work with distances in mm
    data.iloc[:, 1:4] = data.iloc[:, 1:4] * 10
    labels = data.iloc[:, 0].str.lower()

    # Polhemus fiducial points in HEAD space
    nasion = (
        data[labels.str.match("nasion")].iloc[0, 1:4].to_numpy().astype("float64").T
    )
    rpa = (
        data[labels.str.match("right|rpa")].iloc[0, 1:4].to_numpy().astype("float64").T
    )
    lpa = data[labels.str.match("left|lpa")].iloc[0, 1:4].to_numpy().astype("float64").T

    # Polhemus headshape points in HEAD space in mm
    extra = labels == "extra"
    headshape = data[extra] if extra.any() else data[0:num_headshape_pnts]
    headshape = headshape.iloc[:, 1:4].to_numpy().astype("float64").T

    # Save
    os.makedirs(session.coreg_dir, exist_ok=True)
    print(f"Saved: {session.coreg.head_nasion_file}")
    np.savetxt(session.coreg.head_nasion_file, nasion)
    print(f"Saved: {session.coreg.head_rpa_file}")
    np.savetxt(session.coreg.head_rpa_file, rpa)
    print(f"Saved: {session.coreg.head_lpa_file}")
    np.savetxt(session.coreg.head_lpa_file, lpa)
    print(f"Saved: {session.coreg.head_headshape_file}")
    np.savetxt(session.coreg.head_headshape_file, headshape)


def extract_fiducials_and_headshape_from_elc(
    session: Session,
    remove_nose: bool = True,
) -> None:
    """Extract fiducials and headshape points from an ELC file.

    Reads fiducial locations (nasion, LPA, RPA) and headshape points from
    an ELC file (ANT Neuro) and saves them in the RHINO file format.

    The ELC file is expected to contain a ``Positions`` section with three
    fiducial rows (nasion, LPA, RPA) and a ``HeadShapePoints`` section
    with 3D coordinates. All values are assumed to be in mm.

    Parameters
    ----------
    session : Session
        Files of the session. ``session.elc_file`` must be set.
    remove_nose : bool, optional
        Remove headshape points on the nose? A point is considered to be
        on the nose if it is anterior to both LPA and RPA and inferior to
        the nasion.
    """
    if session.elc_file is None:
        raise ValueError("elc_file must have been passed to Session")

    print(f"Saving fiducials/headshape points from {session.elc_file}")

    with open(session.elc_file, "r") as f:
        lines = f.readlines()

    # Extract fiducials from the Positions section
    for i in range(len(lines)):
        if lines[i] == "Positions\n":
            try:
                nasion = np.array(lines[i + 1].split()[-3:]).astype(np.float64)
                lpa = np.array(lines[i + 2].split()[-3:]).astype(np.float64)
                rpa = np.array(lines[i + 3].split()[-3:]).astype(np.float64)
            except ValueError:
                raise ValueError(
                    f"{session.elc_file} must have three fiducials (nasion, LPA, "
                    "RPA) in its Positions section."
                )
            break

    # Extract headshape points from the HeadShapePoints section
    for i in range(len(lines)):
        if lines[i] == "HeadShapePoints\n":
            headshape = (
                np.array([line.split() for line in lines[i + 1 :]]).astype(np.float64).T
            )
            break

    # Optionally remove headshape points on the nose
    if remove_nose:
        on_nose = np.logical_and(
            headshape[0] > max(lpa[0], rpa[0]),
            headshape[2] < nasion[2],
        )
        headshape = headshape[:, ~on_nose]

    # Save
    os.makedirs(session.coreg_dir, exist_ok=True)
    print(f"Saved: {session.coreg.head_nasion_file}")
    np.savetxt(session.coreg.head_nasion_file, nasion)
    print(f"Saved: {session.coreg.head_rpa_file}")
    np.savetxt(session.coreg.head_rpa_file, rpa)
    print(f"Saved: {session.coreg.head_lpa_file}")
    np.savetxt(session.coreg.head_lpa_file, lpa)
    print(f"Saved: {session.coreg.head_headshape_file}")
    np.savetxt(session.coreg.head_headshape_file, headshape)


def remove_stray_headshape_points(
    session: Session,
    nose: bool = True,
    neck: bool = True,
    far: bool = True,
) -> None:
    """Remove stray headshape points.

    Removes headshape points near the nose, on the neck or far away from the
    head. The filtering is done in a coordinate system derived from the
    fiducial points (nasion, LPA, RPA), so it works regardless of the
    original coordinate convention (e.g. Elekta/FIF or CTF/.pos).

    The fiducial-derived axes are:

    - **left-right**: LPA to RPA direction.
    - **anterior**: perpendicular to the left-right axis, pointing towards
      the nasion.
    - **superior**: cross product of the above two axes.

    Parameters
    ----------
    session : Session
        Files of the session.
    nose : bool, optional
        Should we remove headshape points near the nose? Useful to remove
        these if we have defaced structurals or aren't extracting the nose
        from the structural.
    neck : bool, optional
        Should we remove headshape points on the neck?
    far : bool, optional
        Should we remove headshape points far from the head, i.e. outside
        the left-right extent of the fiducials or in front of the nasion?
        Worth turning off for sparse headshapes, where it can remove
        scalp points that are useful for the coregistration.
    """
    print()
    print("Removing stray headshape points")
    print("-------------------------------")

    coreg = session.coreg

    # Load saved headshape and fiducial files
    hs = np.loadtxt(coreg.head_headshape_file)
    nas = np.loadtxt(coreg.head_nasion_file)
    lpa = np.loadtxt(coreg.head_lpa_file)
    rpa = np.loadtxt(coreg.head_rpa_file)

    # Check the headshape array is 2D (3, n_points)
    if hs.ndim != 2 or hs.shape[0] != 3:
        warnings.warn(
            f"Headshape file {coreg.head_headshape_file} has unexpected shape "
            f"{hs.shape}, skipping stray point removal."
        )
        return

    # Build a fiducial-derived coordinate system so the filtering logic
    # is independent of the original axis convention.
    #
    # Origin: midpoint of LPA and RPA
    # Axis 0 (left-right): LPA -> RPA
    # Axis 1 (anterior):   perpendicular, towards nasion
    # Axis 2 (superior):   cross product of the above
    origin = (lpa + rpa) / 2.0
    ax_lr = rpa - lpa
    ax_lr = ax_lr / np.linalg.norm(ax_lr)

    nas_vec = nas - origin
    ax_ant = nas_vec - np.dot(nas_vec, ax_lr) * ax_lr
    ax_ant = ax_ant / np.linalg.norm(ax_ant)

    ax_sup = np.cross(ax_lr, ax_ant)

    # Project everything into the fiducial coordinate system
    def _project(pts: np.ndarray) -> np.ndarray:
        centred = pts - origin[:, np.newaxis]
        return np.array(
            [
                ax_lr @ centred,
                ax_ant @ centred,
                ax_sup @ centred,
            ]
        )

    hs_proj = _project(hs)
    nas_proj = _project(nas[:, np.newaxis]).ravel()
    lpa_proj = _project(lpa[:, np.newaxis]).ravel()
    rpa_proj = _project(rpa[:, np.newaxis]).ravel()

    # Apply the same filtering logic as before, now in the canonical axes
    if nose:
        # Remove headshape points on the nose
        remove = np.logical_and(
            hs_proj[1] > max(lpa_proj[1], rpa_proj[1]),
            hs_proj[2] < nas_proj[2],
        )
        hs = hs[:, ~remove]
        hs_proj = hs_proj[:, ~remove]

    if neck:
        # Remove headshape points on the neck
        remove = hs_proj[2] < min(lpa_proj[2], rpa_proj[2]) - 4
        hs = hs[:, ~remove]
        hs_proj = hs_proj[:, ~remove]

    if far:
        # Remove headshape points far from the head in any direction
        remove = np.logical_or(
            hs_proj[0] < lpa_proj[0] - 5,
            np.logical_or(
                hs_proj[0] > rpa_proj[0] + 5,
                hs_proj[1] > nas_proj[1] + 5,
            ),
        )
        hs = hs[:, ~remove]

    # Overwrite headshape file
    print(f"Overwriting: {coreg.head_headshape_file}")
    np.savetxt(coreg.head_headshape_file, hs)


def delete_headshape_points(
    session: Session | None = None, headshape_file: str | None = None
) -> None:
    """Interactively delete headshape points.

    Shows the headshape points in a 3D figure. Click a point to delete it,
    press ``w`` to write the remaining points back to the file, ``q`` to
    close the figure.

    Parameters
    ----------
    session : Session, optional
        Files of the session; the points come from
        ``session.coreg.head_headshape_file``.
    headshape_file : str, optional
        A headshape text file instead, a (3, n_points) array in mm.
    """
    import sys

    if session is not None:
        headshape_file = session.coreg.head_headshape_file
    if headshape_file is None:
        raise ValueError("Pass a session or a headshape file.")
    points = np.loadtxt(headshape_file)
    x, y, z = (list(points[i]) for i in range(3))
    print(
        f"{len(x)} headshape points. Click a point to delete it, press w to save, q to close."
    )
    sys.stdout.flush()

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    def draw():
        ax.scatter(x, y, z, color="red", marker="o", s=8, alpha=0.7, picker=5)
        plt.draw()

    def on_click(event):
        i = event.ind[0]
        print(f"Deleted {x[i]:.1f}, {y[i]:.1f}, {z[i]:.1f}")
        for coords in (x, y, z):
            coords.pop(i)
        ax.cla()
        draw()

    def on_press(event):
        if event.key == "w":
            np.savetxt(headshape_file, np.array([x, y, z]))
            print(f"{len(x)} points saved to {headshape_file}")
        elif event.key == "q":
            plt.close(fig)

    draw()
    fig.canvas.mpl_connect("pick_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_press)
    plt.show()


def _extract_headshape(
    preproc_file: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract headshape points and fiducials from a FIF file.

    Parameters
    ----------
    preproc_file : str
        Path to FIF file.

    Returns
    -------
    headshape_mm : numpy.ndarray
        3 x N array of headshape points in mm (HEAD space).
    nasion_mm : numpy.ndarray
        Nasion position in mm (HEAD space).
    rpa_mm : numpy.ndarray
        RPA position in mm (HEAD space).
    lpa_mm : numpy.ndarray
        LPA position in mm (HEAD space).
    """
    info = mne.io.read_info(preproc_file)

    headshape_m = []
    nasion_m = None
    rpa_m = None
    lpa_m = None

    for dig in info["dig"]:
        if dig["kind"] == FIFF.FIFFV_POINT_EXTRA:
            headshape_m.append(dig["r"])
        elif dig["kind"] == FIFF.FIFFV_POINT_CARDINAL:
            if dig["ident"] == FIFF.FIFFV_POINT_NASION:
                nasion_m = dig["r"]
            elif dig["ident"] == FIFF.FIFFV_POINT_LPA:
                lpa_m = dig["r"]
            elif dig["ident"] == FIFF.FIFFV_POINT_RPA:
                rpa_m = dig["r"]

    if not headshape_m:
        raise ValueError("No headshape points found in FIF file")
    if nasion_m is None or rpa_m is None or lpa_m is None:
        raise ValueError("Fiducials (nasion, LPA, RPA) not found in FIF file")

    headshape_mm = np.array(headshape_m).T * 1000
    nasion_mm = np.array(nasion_m) * 1000
    rpa_mm = np.array(rpa_m) * 1000
    lpa_mm = np.array(lpa_m) * 1000

    # Remove points below the ears (neck/body)
    z_min = min(lpa_mm[2], rpa_mm[2])
    keep = headshape_mm[2] >= z_min
    headshape_mm = headshape_mm[:, keep]

    return headshape_mm, nasion_mm, rpa_mm, lpa_mm
