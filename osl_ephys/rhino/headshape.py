"""Fiducials and headshape points: extraction from fif, Polhemus and Zebris files, and cleaning them up."""

from __future__ import annotations

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne
from mne.transforms import apply_trans, get_ras_to_neuromag_trans
from osl_ephys.session import Session


def extract_fiducials_and_headshape_from_fif(
    session: Session,
    include_eeg_as_headshape: bool = False,
    include_hpi_as_headshape: bool = True,
    include_extra_as_headshape: bool = True,
) -> None:
    """Extract the fiducials and headshape points from a FIF file.

    The digitised points in session.preproc_file are saved (in mm) to the
    files RHINO uses for coregistration. For CTF data, the points are
    converted to the CTF head frame, which RHINO uses for CTF data.

    Parameters
    ----------
    session : Session
        Files of the session.
    include_eeg_as_headshape : bool, optional
        Should we include EEG locations as headshape points? Only use this if
        the EEG locations were digitised (not a template montage).
    include_hpi_as_headshape : bool, optional
        Should we include HPI locations as headshape points?
    include_extra_as_headshape : bool, optional
        Should we include the extra locations as headshape points?
        These are the unnamed points digitised over the scalp.
    """
    print()
    print("Extracting fiducials/headshape points from fif info")
    print("---------------------------------------------------")

    headshape, nasion, rpa, lpa = _read_fiducials_and_headshape(
        session.preproc_file,
        extra=include_extra_as_headshape,
        eeg=include_eeg_as_headshape,
        hpi=include_hpi_as_headshape,
    )

    # For CTF data, convert the points from MNE's (Neuromag) head frame to the
    # CTF head frame, which RHINO uses for CTF data
    info = mne.io.read_info(session.preproc_file)
    ctf = info["dev_ctf_t"] is not None
    if ctf:
        print("Detected CTF data, converting points to the CTF head frame")
        head_ctf_t = info["dev_ctf_t"]["trans"] @ np.linalg.inv(
            info["dev_head_t"]["trans"]
        )
        nasion, rpa, lpa = apply_trans(head_ctf_t, [nasion, rpa, lpa])
        headshape = apply_trans(head_ctf_t, headshape)

    # Dummy point for CTF data without headshape points
    dummy_headshape = ctf and len(headshape) == 0
    if dummy_headshape:
        headshape = np.zeros((1, 3))

    os.makedirs(session.coreg_dir, exist_ok=True)
    for name, points in [
        ("nasion", nasion),
        ("rpa", rpa),
        ("lpa", lpa),
        ("headshape", headshape.T),
    ]:
        filename = getattr(session.coreg, f"head_{name}_file")
        np.savetxt(filename, points * 1000)
        print(f"Saved: {filename}")

    if dummy_headshape:
        print(
            "Dummy headshape points saved, overwrite "
            f"{session.coreg.head_headshape_file} "
            "or set use_headshape=False in coregister_head_and_mri."
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

    The points are converted to the head frame defined by the fiducials:
    the CTF head frame for CTF data, otherwise MNE's (Neuromag) head frame.
    ``session.preproc_file`` is read to find the system.

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

    with open(session.elc_file) as f:
        lines = [line.strip() for line in f]
    for section in ["Positions", "HeadShapePoints"]:
        if section not in lines:
            raise ValueError(f"{session.elc_file} has no {section} section.")

    # Fiducials: the first three rows of the Positions section
    i = lines.index("Positions")
    rows = [line.split()[-3:] for line in lines[i + 1 : i + 4]]
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError(
            f"{session.elc_file} must have three fiducials (nasion, LPA, RPA) "
            "in its Positions section."
        )
    nasion, lpa, rpa = np.array(rows, dtype=np.float64)

    # Headshape points: the rows after HeadShapePoints
    i = lines.index("HeadShapePoints")
    rows = [line.split() for line in lines[i + 1 :] if line]
    if not rows or any(len(row) != 3 for row in rows):
        raise ValueError(
            f"The HeadShapePoints section of {session.elc_file} must have rows "
            "of x, y, z coordinates."
        )
    headshape = np.array(rows, dtype=np.float64)

    # Convert the points to the CTF head frame defined by the fiducials
    elc_ctf_t = _ctf_head_trans(nasion, lpa, rpa)
    nasion, lpa, rpa = apply_trans(elc_ctf_t, [nasion, lpa, rpa])
    headshape = apply_trans(elc_ctf_t, headshape)

    # Optionally remove headshape points on the nose
    if remove_nose:
        anterior = headshape[:, 0] > max(lpa[0], rpa[0])
        inferior = headshape[:, 2] < nasion[2]
        headshape = headshape[~(anterior & inferior)]

    # For non-CTF data, convert the points to MNE's (Neuromag) head frame
    info = mne.io.read_info(session.preproc_file)
    if info["dev_ctf_t"] is None:
        ctf_head_t = get_ras_to_neuromag_trans(nasion, lpa, rpa)
        nasion, lpa, rpa = apply_trans(ctf_head_t, [nasion, lpa, rpa])
        headshape = apply_trans(ctf_head_t, headshape)

    os.makedirs(session.coreg_dir, exist_ok=True)
    for name, points in [
        ("nasion", nasion),
        ("rpa", rpa),
        ("lpa", lpa),
        ("headshape", headshape.T),
    ]:
        filename = getattr(session.coreg, f"head_{name}_file")
        np.savetxt(filename, points)
        print(f"Saved: {filename}")


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


def _read_fiducials_and_headshape(
    fif_file: str, extra: bool = True, eeg: bool = False, hpi: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read the fiducials and headshape points from a FIF file.

    Parameters
    ----------
    fif_file : str
        Path to FIF file.
    extra : bool, optional
        Include the extra points (digitised over the scalp) as headshape points?
    eeg : bool, optional
        Include the EEG electrode positions as headshape points?
    hpi : bool, optional
        Include the HPI coil positions as headshape points?

    Returns
    -------
    headshape : np.ndarray
        (n_points, 3) headshape points in HEAD space in m. The HPI, EEG and
        extra points, in that order.
    nasion : np.ndarray
        Nasion in HEAD space in m.
    rpa : np.ndarray
        RPA in HEAD space in m.
    lpa : np.ndarray
        LPA in HEAD space in m.
    """
    pos = mne.channels.read_dig_fif(fif_file, verbose="error").get_positions()
    if pos["coord_frame"] != "head":
        raise ValueError(f"The digitised points in {fif_file} are not in HEAD space.")
    if any(pos[f] is None for f in ("nasion", "rpa", "lpa")):
        raise ValueError(f"Fiducials (nasion, LPA, RPA) not found in {fif_file}.")
    points = []
    if hpi and pos["hpi"] is not None:
        points.append(pos["hpi"])
    if eeg and pos["ch_pos"]:
        points.append(np.array(list(pos["ch_pos"].values())))
    if extra and pos["hsp"] is not None:
        points.append(pos["hsp"])
    headshape = np.concatenate(points) if points else np.zeros((0, 3))
    return headshape, pos["nasion"], pos["rpa"], pos["lpa"]


def _ctf_head_trans(nasion: np.ndarray, lpa: np.ndarray, rpa: np.ndarray) -> np.ndarray:
    """Transform to the CTF head frame defined by the fiducials.

    The origin is midway between the LPA and RPA, x points to the nasion, y
    points to the LPA (orthogonal to x) and z points up.

    Parameters
    ----------
    nasion : np.ndarray
        Nasion.
    lpa : np.ndarray
        LPA.
    rpa : np.ndarray
        RPA.

    Returns
    -------
    trans : np.ndarray
        (4, 4) affine transform.
    """
    origin = (lpa + rpa) / 2
    x = nasion - origin
    x /= np.linalg.norm(x)
    y = lpa - origin
    y -= (y @ x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    trans = np.eye(4)
    trans[:3, :3] = [x, y, z]
    trans[:3, 3] = -trans[:3, :3] @ origin
    return trans
