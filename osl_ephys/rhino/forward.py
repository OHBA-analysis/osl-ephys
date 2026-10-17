"""The forward model on the MNI grid: BEM and source space from the RHINO surfaces."""

from __future__ import annotations

import os
import shutil
import subprocess
import numpy as np
import nibabel as nib
import nilearn as nil
import mne
from mne.transforms import Transform, read_trans
from mne.io.constants import FIFF
from osl_ephys import files
from osl_ephys.session import Session, SurfaceFilenames, check_up_to_date, save_params
from osl_ephys.rhino._geometry import _xform_points


def forward_model(
    session: Session,
    model: str = "Single Layer",
    gridstep: int = 8,
    mindist: float = 4.0,
    exclude: float = 0.0,
    eeg: bool = False,
    meg: bool = True,
    verbose: bool = False,
) -> None:
    """Compute forward model.

    Parameters
    ----------
    session : Session
        Files of the session.
    model : string, optional
        Options are:

        - 'Single Layer' to use single layer (brain/cortex).
          Recommended for MEG.
        - 'Triple Layer' to three layers (scalp, inner skull, brain/cortex).
          Recommended for EEG.

    gridstep : int, optional
        Resolution of the dipole grid in mm. A dipole is placed at each voxel
        of the MNI152 brain mask with this resolution (transformed into the
        subject's MRI space), so the dipoles are at the same MNI coordinates
        for every subject and match the voxels of the parcellation files.
        The MNI grid is saved to :code:`session.mni_grid_file`.
    mindist : float
        Exclude points closer than this distance (mm) to the bounding surface.
    exclude : float, optional
        Exclude points closer than this distance (mm) from the center of mass of
        the bounding surface.
    eeg : bool, optional
        Whether to compute forward model for EEG sensors.
    meg : bool, optional
        Whether to compute forward model for MEG sensors.
    """
    print()
    print("Calculating forward model")
    print("-------------------------")

    if model not in ["Single Layer", "Triple Layer"]:
        raise ValueError(f"{model} is an invalid model choice")

    check_up_to_date(session, "coregistration")
    os.makedirs(session.bem_dir, exist_ok=True)
    os.makedirs(session.coreg_dir, exist_ok=True)

    _write_bem_surfaces(session)
    grid_file = f"{session.coreg_dir}/tmp-mni-grid.nii.gz"
    _make_mni_grid(gridstep, grid_file)
    src = _setup_mni_grid_source_space(session, grid_file, exclude=exclude)

    # Save the BEM solution so we can compute the forward model for other
    # dipole locations later, see source_recon.virtual_electrodes
    bem = _make_bem_solution(session, model, verbose=verbose)
    mne.write_bem_solution(
        session.bem_solution_file, bem, overwrite=True, verbose=verbose
    )

    info, head_mri_t = _read_head_model(session)
    fwd = _make_fwd_solution(
        info,
        head_mri_t,
        src=src,
        bem=bem,
        eeg=eeg,
        meg=meg,
        mindist=mindist,
        ignore_ref=True,
        verbose=verbose,
    )
    print(f"{fwd['nsource']} dipoles inside the inner skull")
    mne.write_forward_solution(session.fwd_model_file, fwd, overwrite=True)
    os.replace(grid_file, session.mni_grid_file)

    # Keep what the forward model was computed with next to it
    save_params(
        session.fwd_model_params_file,
        dict(
            model=model,
            gridstep=gridstep,
            mindist=mindist,
            exclude=exclude,
            eeg=eeg,
            meg=meg,
            n_sources=int(fwd["nsource"]),
            nonlinear_registration=os.path.exists(session.surfaces.mri2mni_warp_file),
            surfaces_dir=session.surfaces_dir,
        ),
    )

    print("Forward model complete.")


def _make_bem_solution(
    session: Session, model: str, verbose: bool = False
) -> mne.bem.ConductorModel:
    """Compute the BEM solution.

    Parameters
    ----------
    session : Session
        Files of the session.
    model : str
        'Single Layer' or 'Triple Layer'.
    verbose : bool, optional
        Should we print info to the screen?

    Returns
    -------
    bem : mne.bem.ConductorModel
        BEM solution.
    """
    if model == "Single Layer":
        conductivity = (0.3,)  # for single layer
    elif model == "Triple Layer":
        conductivity = (0.3, 0.006, 0.3)  # for three layers
    else:
        raise ValueError(f"{model} is an invalid model choice")

    # The BEM solution requires a BEM model which describes the geometry of the
    # head the conductivities of the different tissues.
    # See: https://mne.tools/stable/auto_tutorials/forward/30_forward.html#sphx-glr-auto-tutorials-forward-30-forward-py
    #
    # Note that the BEM does not involve any use of transforms between spaces.
    # The BEM only depends on the head geometry and conductivities.
    # It is therefore independent from the MEG data and the head position.
    #
    # This will get the surfaces from: subjects_dir/subject/bem/inner_skull.surf,
    # which is where _write_bem_surfaces will have put it.
    bem_model = mne.make_bem_model(
        subjects_dir=session.outdir,
        subject=session.head_model_id,
        ico=None,
        conductivity=conductivity,
        verbose=verbose,
    )
    return mne.make_bem_solution(bem_model, verbose=verbose)


def _write_bem_surfaces(session: Session) -> None:
    """Copy the BET surfaces to where MNE expects them.

    This copies the CoregFilenames.bet_*_surf_file files to
    `subjects_dir/subject/bem/*.surf` since this is where mne expects to find
    them when mne.make_bem_model is called.

    Parameters
    ----------
    session : Session
        Files of the session.
    """
    # Note that due to the unusual naming conventions used by BET and MNE:
    # - bet_inskull_*_file is actually the brain surface
    # - bet_outskull_*_file is actually the inner skull surface
    # - bet_outskin_*_file is the outer skin/scalp surface
    #
    # These correspond in mne to (in order):
    # - inner_skull
    # - outer_skull
    # - outer_skin
    #
    # This means that for single shell model, i.e. with conductivities set to
    # length one, the surface used by MNE will always be the inner_skull,
    # i.e. it actually corresponds to the brain/cortex surface!! Not sure that
    # is correct/optimal.
    #
    # Note that this is done in Fieldtrip too!, see the "Realistic single-shell
    # model, using brain surface from segmented mri" section at:
    # https://www.fieldtriptoolbox.org/example/make_leadfields_using_different_headmodels/#realistic-single-shell-model-using-brain-surface-from-segmented-mri
    #
    # However, others are clear that it should really be the actual inner
    # surface of the skull, see the "single-shell Boundary Element Model (BEM)"
    # bit at: https://imaging.mrc-cbu.cam.ac.uk/meg/SpmForwardModels

    # -------------------------------------------------------------------
    # Move the surfaces to where MNE expects to find them for the forward
    # modelling, see make_bem_model in mne/bem.py
    # -------------------------------------------------------------------

    # Note that the coreg surf files are in scaled MRI space
    verts, tris = mne.surface.read_surface(session.coreg.bet_inskull_surf_file)
    tris = tris.astype(int)
    mne.surface.write_surface(
        f"{session.bem_dir}/inner_skull.surf",
        verts,
        tris,
        file_format="freesurfer",
        overwrite=True,
    )
    print("Using bet_inskull_surf_file for single shell surface")

    verts, tris = mne.surface.read_surface(session.coreg.bet_outskull_surf_file)
    tris = tris.astype(int)
    mne.surface.write_surface(
        f"{session.bem_dir}/outer_skull.surf",
        verts,
        tris,
        file_format="freesurfer",
        overwrite=True,
    )

    verts, tris = mne.surface.read_surface(session.coreg.bet_outskin_surf_file)
    tris = tris.astype(int)
    mne.surface.write_surface(
        f"{session.bem_dir}/outer_skin.surf",
        verts,
        tris,
        file_format="freesurfer",
        overwrite=True,
    )


def _setup_mni_grid_source_space(
    session: Session, grid_file: str, exclude: float = 0.0
) -> mne.SourceSpaces:
    """Set up a source space with a dipole at each voxel of an MNI grid.

    The voxels of the grid are transformed into the subject's (scaled) MRI
    space, so every subject has a dipole at the same MNI coordinates. Dipoles
    outside the inner skull surface (or closer than mindist to it) are
    excluded when the forward model is computed.

    Parameters
    ----------
    session : Session
        Files of the session.
    grid_file : str
        MNI grid (mask) file, see :func:`_make_mni_grid`.
    exclude : float, optional
        Exclude points closer than this distance (mm) from the center of mass
        of the inner skull surface.

    Returns
    -------
    src : mne.SourceSpaces
        A single discrete source space in MRI space. The order of the dipoles
        is the order of the voxels returned by :code:`_mni_grid_coords`.
    """
    coords_mni = _mni_grid_coords(grid_file)
    print(f"MNI grid: {len(coords_mni)} voxels")
    src = _mni_source_space(session, coords_mni)
    if exclude > 0:
        # Exclude points close to the centre of mass of the inner skull
        surf = mne.surface.read_surface(
            f"{session.bem_dir}/inner_skull.surf", return_dict=True
        )[-1]
        dist = np.linalg.norm(src[0]["rr"] * 1000 - surf["rr"].mean(axis=0), axis=1)
        src[0]["inuse"][dist < exclude] = 0
        src[0]["vertno"] = np.flatnonzero(src[0]["inuse"])
        src[0]["nuse"] = len(src[0]["vertno"])
    return src


def _mni_source_space(session: Session, coords_mni: np.ndarray) -> mne.SourceSpaces:
    """Discrete source space with a dipole at each MNI coordinate.

    Parameters
    ----------
    session : Session
        Files of the session.
    coords_mni : np.ndarray
        (n, 3) coordinates in MNI space in mm.

    Returns
    -------
    src : mne.SourceSpaces
        Source space in scaled MRI space (in metres), which is what
        _make_fwd_solution expects.
    """
    coords_mri = _mni_to_scaledmri(session, coords_mni)
    nn = np.tile([0.0, 0.0, 1.0], (len(coords_mri), 1))
    return mne.setup_volume_source_space(
        pos=dict(rr=coords_mri / 1000, nn=nn), verbose=False
    )


def _make_mni_grid(gridstep: int, grid_file: str) -> None:
    """Save the MNI152 brain mask at a resolution of gridstep.

    The masks in osl-files share the origin of the MNI152 template, so the
    voxels of the grid line up with those of the parcellation files. A
    resolution that osl-files does not have is made by resampling the 1 mm
    mask onto a grid with the same origin.

    Parameters
    ----------
    gridstep : int
        Resolution in mm.
    grid_file : str
        Output file.
    """
    try:
        shutil.copyfile(
            files.mask.file(f"MNI152_T1_{int(gridstep)}mm_brain.nii.gz"), grid_file
        )
    except FileNotFoundError:
        img = nib.load(files.mask.file("MNI152_T1_1mm_brain.nii.gz"))
        affine = np.diag([-gridstep, gridstep, gridstep, 1.0])
        affine[:3, 3] = img.affine[:3, 3]
        shape = np.ceil(np.array(img.shape) / gridstep).astype(int)
        grid = nil.image.resample_img(
            img, affine, shape, interpolation="nearest", force_resample=True
        )
        nib.save(nil.image.binarize_img(grid, 0, copy_header=True), grid_file)


def _mni_grid_coords(grid_file: str) -> np.ndarray:
    """MNI coordinates of the voxels of an MNI grid (mask) file.

    Parameters
    ----------
    grid_file : str
        Path to the mask file.

    Returns
    -------
    coords : np.ndarray
        (n_voxels, 3) coordinates in mm of the non-zero voxels.
    """
    img = nib.load(grid_file)
    ijk = np.array(np.where(np.asanyarray(img.dataobj) != 0)).T
    return nib.affines.apply_affine(img.header.get_sform(), ijk)


def _mni_to_mri(surfaces: SurfaceFilenames, coords_mni: np.ndarray) -> np.ndarray:
    """Transform points from MNI space to (unscaled) MRI space.

    Uses the nonlinear registration (FNIRT warp) if extract_surfaces was
    called with nonlinear_registration=True, otherwise the FLIRT affine
    registration.

    Parameters
    ----------
    surfaces : SurfaceFilenames
        Surface extraction file paths.
    coords_mni : np.ndarray
        (n, 3) coordinates in MNI space in mm.

    Returns
    -------
    coords_mri : np.ndarray
        (n, 3) coordinates in MRI space in mm.
    """
    if not os.path.exists(surfaces.mri2mni_warp_file):
        mni_mri_t = read_trans(surfaces.mni_mri_t_file)["trans"]
        return _xform_points(mni_mri_t, coords_mni.T).T

    # std2imgcoord stops reading at a coordinate that repeats the one before
    # it, so each coordinate is passed once
    lines = [" ".join(f"{c:.6f}" for c in xyz) for xyz in coords_mni]
    unique_lines, inverse = np.unique(lines, return_inverse=True)

    # Command: std2imgcoord -img <mri_file> -std <MNI152_T1_2mm> \
    #          -warp <mri2mni_warp_file> -mm -
    result = subprocess.run(
        [
            f"{surfaces.fsl_dir}/bin/std2imgcoord",
            *["-img", surfaces.mri_file, "-std", surfaces.std_head_2mm_file],
            *["-warp", surfaces.mri2mni_warp_file, "-mm", "-"],
        ],
        input="\n".join(unique_lines),
        capture_output=True,
        text=True,
    )
    coords_mri = np.array(
        [line.split() for line in result.stdout.strip().splitlines()], dtype=float
    )
    if result.returncode != 0 or coords_mri.shape != (len(unique_lines), 3):
        raise RuntimeError(f"std2imgcoord failed:\n{result.stderr}")
    return coords_mri[inverse]


def _mni_to_scaledmri(session: Session, coords_mni: np.ndarray) -> np.ndarray:
    """Transform points from MNI space to scaled MRI space.

    Parameters
    ----------
    session : Session
        Files of the session.
    coords_mni : np.ndarray
        (n, 3) coordinates in MNI space in mm.

    Returns
    -------
    coords_mri : np.ndarray
        (n, 3) coordinates in scaled MRI space in mm.
    """
    # MNI -> (unscaled) MRI -> head -> scaled MRI
    coords_mri = _mni_to_mri(session.surfaces, coords_mni)
    head_mri_t = read_trans(session.coreg.head_mri_t_file)["trans"]
    head_scaledmri_t = read_trans(session.coreg.head_scaledmri_t_file)["trans"]
    xform = head_scaledmri_t @ np.linalg.inv(head_mri_t)
    return _xform_points(xform, coords_mri.T).T


def _read_head_model(session: Session) -> tuple[mne.Info, Transform]:
    """Read the sensor info and head-to-MRI transform of the coregistration.

    Parameters
    ----------
    session : Session
        Files of the session.

    Returns
    -------
    info : mne.Info
        Sensor info saved by the coregistration.
    head_mri_t : mne.transforms.Transform
        Transform from head to (scaled) MRI coordinates, in metres.
    """
    info = mne.io.read_info(session.coreg.info_fif_file)
    head_mri_t = read_trans(session.coreg.head_scaledmri_t_file)
    # RHINO does everything in mm, so need to convert it to metres which is
    # what MNE expects. To change units on an xform, just need to change the
    # translation part and leave the rotation alone
    head_mri_t["trans"][0:3, -1] = head_mri_t["trans"][0:3, -1] / 1000
    return info, head_mri_t


def _make_fwd_solution(
    info: mne.Info,
    head_mri_t: Transform,
    src: mne.SourceSpaces,
    bem: mne.bem.ConductorModel | str,
    meg: bool = True,
    eeg: bool = True,
    mindist: float = 0.0,
    ignore_ref: bool = False,
    verbose: bool | None = None,
) -> mne.Forward:
    """Calculate a forward solution.

    This is a wrapper for mne.make_forward_solution.

    Parameters
    ----------
    info : mne.Info
        Sensor info.
    head_mri_t : mne.transforms.Transform
        Transform from head to MRI coordinates, in metres (see
        :func:`_read_head_model`).
    src : instance of SourceSpaces
        Volumetric source space in MRI coordinates.
    bem : instance of ConductorModel | str
        BEM solution, or the path to one.
    meg : bool, optional
        Include MEG computations?
    eeg : bool, optional
        Include EEG computations?
    mindist : float, optional
        Minimum distance of sources from inner skull surface (in mm).
    ignore_ref : bool, optional
        If True, do not include reference channels in compensation.
        This option should be True for KIT files, since forward computation
        with reference channels is not currently supported.
    verbose : bool, optional
        Should we print info to the screen?

    Returns
    -------
    fwd : instance of Forward
        The forward solution.

    Notes
    -----
    Forward modelling is done in head space.

    The coords of points to reconstruct to can be found in the output here:

    >>> fwd['src'][0]['rr'][fwd['src'][0]['vertno']]

    where they are in head space in metres.

    The same coords of points to reconstruct to can be found in the input here:

    >>> src[0]['rr'][src[0]['vertno']]

    where they are in native MRI space in metres.
    """
    # src should be in MRI space. Let's just check that is the case
    if src[0]["coord_frame"] != FIFF.FIFFV_COORD_MRI:
        raise RuntimeError("src is not in MRI coordinates")

    # Get bem solution
    if isinstance(bem, str):
        bem = mne.read_bem_solution(bem)
    else:
        if not isinstance(bem, mne.bem.ConductorModel):
            raise TypeError("bem must be a string or ConductorModel")
        bem = bem.copy()

    for i in range(len(bem["surfs"])):
        bem["surfs"][i]["tris"] = bem["surfs"][i]["tris"].astype(int)

    # The forward model is done in head space
    fwd = mne.make_forward_solution(
        info,
        trans=head_mri_t,
        src=src,
        bem=bem,
        eeg=eeg,
        meg=meg,
        mindist=mindist,
        ignore_ref=ignore_ref,
        verbose=verbose,
    )

    # fwd should be in Head space. Let's just check that is the case:
    if fwd["src"][0]["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
        raise RuntimeError("fwd['src'][0] is not in HEAD coordinates")

    return fwd
