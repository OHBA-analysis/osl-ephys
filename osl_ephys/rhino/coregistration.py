"""Coregistration of the head (fiducials and headshape points) to the MRI surfaces, its error and its plots."""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from fsl import wrappers as fsl_wrappers
import mne
from mne.transforms import (
    Transform,
    read_trans,
    write_trans,
    invert_transform,
    combine_transforms,
    apply_trans,
    rotation,
    _get_trans,
)
from mne.viz.backends.renderer import _get_renderer
from osl_ephys.session import Session, SurfaceFilenames, save_params
from osl_ephys.rhino._geometry import (
    _get_sform,
    _niimask2indexpointcloud,
    _transform_vtk_mesh,
    _xform_points,
)
from osl_ephys.rhino.headshape import _extract_headshape
from osl_ephys.rhino.surfaces import (
    _create_freesurfer_mesh_from_bet_surface,
    _create_freesurfer_meshes_from_bet_surfaces,
)


def scale_surfaces_to_headshape(
    preproc_file: str,
    surfaces_dir: str,
    outdir: str,
    n_init: int = 10,
) -> str:
    """Scale surfaces to match headshape points.

    Computes an affine transform that scales an existing set of surfaces
    (from ``extract_surfaces`` or the bundled MNI152 surfaces) to match
    the headshape digitisation points in a FIF file. The scaled MRI and
    surfaces are written to ``outdir``.

    This is useful when no individual structural MRI is available (e.g.,
    OPM recordings with EinScan headshape).

    The method:

    1. Aligns the headshape to the template scalp surface using fiducials
       and ICP.
    2. Computes an affine transform (with anisotropic scaling) from
       nearest-neighbour correspondences between the aligned surfaces.
    3. Applies the transform to the MRI and surfaces.

    Parameters
    ----------
    preproc_file : str
        Path to preprocessed FIF file containing headshape points in
        ``info['dig']``.
    surfaces_dir : str
        Path to a directory containing extracted surfaces (from
        ``extract_surfaces``). For the default MNI152 brain, use
        ``osl_ephys.files.mni152_surfaces.directory``.
    outdir : str
        Output directory for the scaled MRI and surfaces. Pass this
        as ``surfaces_dir`` when creating ``Session``.
    n_init : int, optional
        Number of random initialisations for ICP alignment.

    Returns
    -------
    outdir : str
        Path to the output directory containing the scaled MRI and
        surfaces.
    """
    outdir = str(Path(outdir).resolve())
    os.makedirs(outdir, exist_ok=True)

    print()
    print("Scaling surfaces to headshape")
    print("-----------------------------")

    # -------------
    # Load surfaces
    # -------------
    sfns = SurfaceFilenames(surfaces_dir)

    outskin_sform = _get_sform(sfns.bet_outskin_mesh_file)["trans"]
    scalp_voxels = _niimask2indexpointcloud(sfns.bet_outskin_mesh_file)
    mni2mri_mm = read_trans(sfns.mni_mri_t_file)["trans"]

    # Subsample scalp surface for ICP efficiency
    n_scalp = scalp_voxels.shape[1]
    if n_scalp > 5000:
        indices = np.random.choice(n_scalp, 5000, replace=False)
        scalp_voxels = scalp_voxels[:, indices]
    scalp_mm = _xform_points(outskin_sform, scalp_voxels)

    # -------------------------------
    # Extract headshape from FIF file
    # -------------------------------
    print(f"Loading headshape from {preproc_file}")
    headshape_mm, nasion_mm, rpa_mm, lpa_mm = _extract_headshape(preproc_file)
    print(f"Found {headshape_mm.shape[1]} headshape points")

    # ---------------------------------
    # Initial alignment using fiducials
    # ---------------------------------
    print("Computing initial alignment via fiducials")

    mni_nasion = np.array([1.0, 85.0, -41.0])
    mni_rpa = np.array([83.0, -20.0, -65.0])
    mni_lpa = np.array([-83.0, -20.0, -65.0])

    mri_nasion = _xform_points(mni2mri_mm, mni_nasion)
    mri_rpa = _xform_points(mni2mri_mm, mni_rpa)
    mri_lpa = _xform_points(mni2mri_mm, mni_lpa)

    polhemus_fid = np.column_stack(
        [
            nasion_mm.reshape(-1, 1),
            rpa_mm.reshape(-1, 1),
            lpa_mm.reshape(-1, 1),
        ]
    )
    mri_fid = np.column_stack(
        [
            mri_nasion.reshape(-1, 1),
            mri_rpa.reshape(-1, 1),
            mri_lpa.reshape(-1, 1),
        ]
    )
    head2mri_xform, _ = _rigid_transform_3D(mri_fid, polhemus_fid)

    # --------------
    # ICP refinement
    # --------------
    print(f"Running ICP with {n_init} initialisations")

    headshape_mri_mm = _xform_points(head2mri_xform, headshape_mm)
    xform_icp, _, _ = _rhino_icp(scalp_mm, headshape_mri_mm, n_init=n_init)
    head2mri_refined = xform_icp @ head2mri_xform

    # -----------------------------------------
    # Compute affine from point correspondences
    # -----------------------------------------
    print("Computing affine transform from surface correspondences")
    headshape_aligned = _xform_points(head2mri_refined, headshape_mm)

    scalp_tree = KDTree(scalp_mm.T)
    _, nn_indices = scalp_tree.query(headshape_aligned.T)
    template_correspondences = scalp_mm[:, nn_indices]

    n_pts = headshape_aligned.shape[1]
    src = np.vstack([template_correspondences, np.ones(n_pts)])
    dst = headshape_aligned

    affine_warp, _, _, _ = np.linalg.lstsq(src.T, dst.T, rcond=None)
    affine_warp = affine_warp.T
    warp_xform = np.eye(4)
    warp_xform[:3, :] = affine_warp

    print(
        f"  Scaling: x={warp_xform[0,0]:.3f} "
        f"y={warp_xform[1,1]:.3f} z={warp_xform[2,2]:.3f}"
    )

    # ---------------
    # Save scaled MRI
    # ---------------
    print("Saving scaled MRI and surfaces")

    mri_img = nib.load(sfns.mri_file)
    mri_sform = mri_img.header.get_sform()
    sform_code = int(mri_img.header["sform_code"])
    scaled_sform = warp_xform @ mri_sform

    smri_out = os.path.join(outdir, "smri.nii.gz")
    smri_img = nib.Nifti1Image(mri_img.get_fdata(), scaled_sform)
    smri_img.header.set_sform(scaled_sform, code=sform_code)
    nib.save(smri_img, smri_out)

    # --------------------
    # Save scaled surfaces
    # --------------------
    for mesh_name in ["outskin_mesh", "inskull_mesh", "outskull_mesh"]:
        # Scale NIfTI mesh sform
        src_nii = os.path.join(surfaces_dir, f"{mesh_name}.nii.gz")
        dst_nii = os.path.join(outdir, f"{mesh_name}.nii.gz")
        mesh_img = nib.load(src_nii)
        mesh_sform = warp_xform @ mesh_img.header.get_sform()
        new_img = nib.Nifti1Image(mesh_img.get_fdata(), mesh_sform)
        new_img.header.set_sform(mesh_sform, code=sform_code)
        nib.save(new_img, dst_nii)

        # Scale VTK mesh vertex coordinates
        src_vtk = os.path.join(surfaces_dir, f"{mesh_name}.vtk")
        dst_vtk = os.path.join(outdir, f"{mesh_name}.vtk")
        if os.path.exists(src_vtk):
            _transform_vtk_mesh(src_vtk, src_nii, dst_vtk, dst_nii, warp_xform)

    # Save scaled MNI -> MRI transform
    mni_mri_t = warp_xform @ mni2mri_mm
    mni_mri_t_out = Transform("mni_tal", "mri", mni_mri_t)
    write_trans(
        os.path.join(outdir, "mni_mri-trans.fif"),
        mni_mri_t_out,
        overwrite=True,
    )

    print(f"Surfaces saved: {outdir}")
    return outdir


def save_coregistration_files(session: Session) -> None:
    """Data is already coregistered, just save the files needed for RHINO.

    Assumes that the sensor locations and fiducials/headshape points (if
    there are any) are already in MRI space. This means that dev_head_t
    is identity and that dev_mri_t is identity.

    Parameters
    ----------
    session : Session
        Files of the session.
    """

    print()
    print("Running dummy coregistration")
    print("----------------------------")
    os.makedirs(session.coreg_dir, exist_ok=True)

    # Paths to files
    cfns = session.coreg
    sfns = session.surfaces

    # ------------------------------------------
    # Copy fif_file to new file for modification
    # ------------------------------------------

    # Get info from fif file
    info = mne.io.read_info(session.preproc_file)

    raw = mne.io.RawArray(np.zeros([len(info["ch_names"]), 1]), info)
    raw.save(cfns.info_fif_file, overwrite=True)

    # Write native (mri) voxel index to native (mri) transform
    xform_nativeindex2scalednative = _get_sform(sfns.bet_outskin_mesh_file)["trans"]
    mrivoxel_scaledmri_t = Transform(
        "mri_voxel", "mri", np.copy(xform_nativeindex2scalednative)
    )
    write_trans(cfns.mrivoxel_scaledmri_t_file, mrivoxel_scaledmri_t, overwrite=True)

    # head_mri-trans.fif for scaled MRI
    head_mri_t = Transform("head", "mri", np.identity(4))
    write_trans(cfns.head_mri_t_file, head_mri_t, overwrite=True)
    write_trans(cfns.head_scaledmri_t_file, head_mri_t, overwrite=True)

    # Copy meshes to coreg dir from surfaces dir
    for filename in [
        "mri_file",
        "bet_outskin_mesh_file",
        "bet_outskin_plus_nose_mesh_file",
        "bet_inskull_mesh_file",
        "bet_outskull_mesh_file",
        "bet_outskin_mesh_vtk_file",
        "bet_inskull_mesh_vtk_file",
        "bet_outskull_mesh_vtk_file",
    ]:
        src = getattr(sfns, filename)
        dst = getattr(cfns, filename)
        if os.path.exists(src):
            shutil.copyfile(src, dst)

    # ------------------------------------------------------------------------
    # Create sMRI-derived freesurfer meshes in native/mri space in mm, for use
    # by forward modelling
    # ------------------------------------------------------------------------

    nativeindex_scalednative_t = np.copy(xform_nativeindex2scalednative)
    mrivoxel_scaledmri_t = Transform("mri_voxel", "mri", nativeindex_scalednative_t)
    _create_freesurfer_meshes_from_bet_surfaces(cfns, mrivoxel_scaledmri_t["trans"])

    # -----------------------
    # Plot the coregistration
    # -----------------------

    plot_coregistration(
        session,
        display_sensors=False,
        display_fiducials=False,
        display_headshape_pnts=False,
        include_nose=False,
    )

    print("Coregistration complete.")


def coregister_head_and_mri(
    session: Session,
    use_headshape: bool = True,
    use_nose: bool = True,
    allow_mri_scaling: bool = False,
    mni_fiducials: dict[str, np.ndarray] | None = None,
    n_init: int = 1,
    plot_type: str = "png",
    show: bool = False,
) -> None:
    """Coregister HEAD (polhemus) and MRI space.

    Calculates a linear, affine transform from HEAD space to MRI space
    using headshape points (if use_headshape=True).

    Requires ``rhino.extract_surfaces`` to have been run.

    RHINO firsts registers the polhemus-derived fiducials (nasion, rpa, lpa) in
    HEAD space to the MRI-derived fiducials in native MRI space.

    RHINO then refines this by making use of polhemus-derived headshape points
    that trace out the surface of the head (scalp).

    Finally, these polhemus-derived headshape points in HEAD space are
    registered to the MRI-derived scalp surface in native MRI space.

    In more detail:

    1)  Map location of fiducials in MNI standard space brain to native MRI
        space. These are then used as the location of the MRI-derived fiducials
        in native MRI space.

    2a) We have polhemus-derived fiducials in HEAD space and MRI-derived fiducials
        in native MRI space. Use these to estimate the affine xform from native
        MRI space to HEAD space.

    2b) We can also optionally learn the best scaling to add to this affine
        xform, such that the MRI-derived fiducials are scaled in size to better
        match the polhemus-derived fiducials. This assumes that we trust the size
        (e.g. in mm) of the polhemus-derived fiducials, but not the size of
        MRI-derived fiducials. E.g. this might be the case if we do not trust
        the size (e.g. in mm) of the MRI, or if we are using a template MRI
        that would has not come from this subject.

    3)  If a scaling is learnt in step 2, we apply it to MRI, and to anything
        derived from MRI.

    4)  Transform MRI-derived headshape points into HEAD space.

    5)  We have the polhemus-derived headshape points in HEAD space and the
        MRI-derived headshape (scalp surface) in native MRI space.  Use these
        to estimate the affine xform from native MRI space using the ICP
        algorithm initilaised using the xform estimate in step 2.

    Parameters
    ----------
    session : Session
        Files of the session.
    use_headshape : bool, optional
        Determines whether polhemus derived headshape points are used.
    use_nose : bool, optional
        Determines whether nose is used to aid coregistration,
        only relevant if use_headshape=True.
    allow_mri_scaling : bool, optional
        Indicates if we are to allow scaling of the MRI, such that the
        MRI-derived fiducials are scaled in size to better match the
        polhemus-derived fiducials. This assumes that we trust the size
        (e.g. in mm) of the polhemus-derived fiducials, but not the size
        of the MRI-derived fiducials.
        E.g. this might be the case if we do not trust the size (e.g. in mm)
        of the MRI, or if we are using a template MRI that has not come from
        this subject.
    mni_fiducials : list, optional
        Fiducials for the MRI in MNI space. Must be [nasion, rpa, lpa],
        where nasion, rpa, lpa are 3D coordinates.
        Defaults to [[1, 85, -41], [83, -20, -65], [-83, -20, -65]].
    n_init : int, optional
        Number of initialisations for the ICP algorithm that performs coregistration.
    plot_type : str, optional
        Type of coregistration plot to save. Options are "png", "html" or None.
        "png" saves static PNG images (requires a display/render window).
        "html" saves an interactive HTML file (works in headless environments).
        None skips plotting.
    show : bool, optional
        Whether to display the coregistration plot interactively. Default is
        False (suitable for batch processing).
    """

    # Note the jargon used varies for xforms and coord spaces:
    # - MEG (device): dev_head_t --> HEAD (polhemus)
    # - HEAD (polhemus): head_mri_t (polhemus2native) --> MRI (native)
    # - MRI (native): mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices

    # RHINO does everything in mm

    print()
    print("Running coregistration (HEAD (polhemus) -> MRI)")
    print("-----------------------------------------------")
    os.makedirs(session.coreg_dir, exist_ok=True)

    # Paths to files
    cfns = session.coreg
    sfns = session.surfaces

    if not use_headshape:
        use_nose = False

    if use_nose and not os.path.exists(sfns.bet_outskin_plus_nose_mesh_file):
        raise FileNotFoundError(
            f"{sfns.bet_outskin_plus_nose_mesh_file} not found. The surfaces "
            "do not include the nose (rhino.extract_surfaces was run with "
            "include_nose=False, or the MNI152 standard brain is being used). "
            "Pass use_nose=False."
        )

    if use_nose:
        print("The MRI-derived nose is going to be used to aid coregistration.")
        print(
            "Please ensure that rhino.extract_surfaces was run with include_nose=True."
        )
        print("Please ensure that the headshape points include the nose.")
    else:
        print("The MRI-derived nose is not going to be used to aid coregistration.")
        print("Please ensure that the headshape points do not include the nose")

    # Copy fif_file to new file for modification
    # and change dev_head_t to equal dev_ctf_t in fif file info
    info = mne.io.read_info(session.preproc_file)

    dev_ctf_t = info["dev_ctf_t"]
    if dev_ctf_t is not None:
        print("Detected CTF data")
        print("Setting dev_head_t equal to dev_ctf_t in fif file info.")
        dev_head_t, _ = _get_trans(info["dev_head_t"], "meg", "head")
        dev_head_t["trans"] = dev_ctf_t["trans"]

    raw = mne.io.RawArray(np.zeros([len(info["ch_names"]), 1]), info)
    raw.save(cfns.info_fif_file, overwrite=True)

    # Load in the "polhemus-derived fiducials" in HEAD space; the headshape
    # points only when they are used (a session may have none)
    if use_headshape:
        print(f"Loading: {cfns.head_headshape_file}")
        polhemus_headshape_head = np.loadtxt(cfns.head_headshape_file)
    else:
        polhemus_headshape_head = np.zeros((3, 0))

    print(f"Loading: {cfns.head_nasion_file}")
    polhemus_nasion_head = np.loadtxt(cfns.head_nasion_file)

    print(f"Loading: {cfns.head_rpa_file}")
    polhemus_rpa_head = np.loadtxt(cfns.head_rpa_file)

    print(f"Loading: {cfns.head_lpa_file}")
    polhemus_lpa_head = np.loadtxt(cfns.head_lpa_file)

    # ----------------------------------------------------------------------
    # 1) Map location of fiducials in MNI standard space brain to native
    #    MRI space. These are then used as the location of the MRI-derived
    #    fiducials in native MRI space.
    # ----------------------------------------------------------------------

    if mni_fiducials is None:
        # Known locations of MNI derived fiducials in MNI coords
        print("Using known MNI-derived fiducials")
        mni_fiducials = [[1, 85, -41], [83, -20, -65], [-83, -20, -65]]

    mni_nasion_mni = np.asarray(mni_fiducials[0])
    mni_rpa_mni = np.asarray(mni_fiducials[1])
    mni_lpa_mni = np.asarray(mni_fiducials[2])

    mni_mri_t = read_trans(sfns.mni_mri_t_file)

    # Apply this xform to the MNI fiducials to get what we call the
    # "MRI-derived fiducials" in native space
    mri_nasion_native = _xform_points(mni_mri_t["trans"], mni_nasion_mni)
    mri_lpa_native = _xform_points(mni_mri_t["trans"], mni_lpa_mni)
    mri_rpa_native = _xform_points(mni_mri_t["trans"], mni_rpa_mni)

    # ----------------------------------------------------------------------
    # 2a) We have polhemus-derived fiducials in polhemus space and MRI-derived
    #     fiducials in native MRI space. Use these to estimate the affine xform
    #     from native MRI space to polhemus (head) space.
    #
    # 2b) We can also optionally learn the best scaling to add to this
    #     affine xform, such that the MRI-derived fiducials are scaled in size
    #     to better match the polhemus-derived fiducials. This assumes that we
    #     trust the size (e.g. in mm) of the polhemus-derived fiducials, but not
    #     the size of the MRI-derived fiducials. E.g. this might be the case if
    #     we do not trust the size (e.g. in mm) of the MRI, or if we are
    #     using a template MRI that has not come from this subject.
    # ----------------------------------------------------------------------

    # Note that mri_fid_native are the MRI-derived fiducials in native space
    polhemus_fid_head = np.concatenate(
        (
            np.reshape(polhemus_nasion_head, [-1, 1]),
            np.reshape(polhemus_rpa_head, [-1, 1]),
            np.reshape(polhemus_lpa_head, [-1, 1]),
        ),
        axis=1,
    )
    mri_fid_native = np.concatenate(
        (
            np.reshape(mri_nasion_native, [-1, 1]),
            np.reshape(mri_rpa_native, [-1, 1]),
            np.reshape(mri_lpa_native, [-1, 1]),
        ),
        axis=1,
    )

    # Estimate the affine xform from native MRI space to polhemus (head)
    # space. Optionally includes a scaling of the MRI, captured by
    # xform_native2scalednative
    xform_scalednative2head, xform_native2scalednative = _rigid_transform_3D(
        polhemus_fid_head,
        mri_fid_native,
        compute_scaling=allow_mri_scaling,
    )

    # ----------------------------------------------------------------------
    # 3) Apply scaling from xform_native2scalednative to MRI, and to stuff
    #    derived from MRI, including:
    #    - MRI
    #    - MRI-derived surfaces
    #    - MRI-derived fiducials
    # ----------------------------------------------------------------------

    # Scale MRI and MRI-derived mesh files by changing their sform
    xform_nativeindex2native = _get_sform(sfns.mri_file)["trans"]
    xform_nativeindex2scalednative = (
        xform_native2scalednative @ xform_nativeindex2native
    )
    filenames = [
        "mri_file",
        "bet_outskin_mesh_file",
        "bet_inskull_mesh_file",
        "bet_outskull_mesh_file",
    ]
    if use_nose:
        filenames.append("bet_outskin_plus_nose_mesh_file")
    for filename in filenames:
        shutil.copyfile(getattr(sfns, filename), getattr(cfns, filename))
        # Command: fslorient -setsform <sform> <mri_file>
        sform = xform_nativeindex2scalednative.flatten()
        fsl_wrappers.misc.fslorient(getattr(cfns, filename), setsform=tuple(sform))

    # Scale vtk meshes
    for mesh_fname, vtk_fname in zip(
        [
            "bet_outskin_mesh_file",
            "bet_inskull_mesh_file",
            "bet_outskull_mesh_file",
        ],
        [
            "bet_outskin_mesh_vtk_file",
            "bet_inskull_mesh_vtk_file",
            "bet_outskull_mesh_vtk_file",
        ],
    ):
        _transform_vtk_mesh(
            getattr(sfns, vtk_fname),
            getattr(sfns, mesh_fname),
            getattr(cfns, vtk_fname),
            getattr(cfns, mesh_fname),
            xform_native2scalednative,
        )

    # Put MRI-derived fiducials into scaled MRI space
    xform = xform_native2scalednative @ mni_mri_t["trans"]
    mri_nasion_scalednative = _xform_points(xform, mni_nasion_mni)
    mri_lpa_scalednative = _xform_points(xform, mni_lpa_mni)
    mri_rpa_scalednative = _xform_points(xform, mni_rpa_mni)

    # --------------------------------------------------------------------
    # 4) Now we can transform MRI-derived headshape points into HEAD space
    # --------------------------------------------------------------------

    # File containing the "MRI-derived headshape points"
    if use_nose:
        outskin_mesh_file = cfns.bet_outskin_plus_nose_mesh_file
    else:
        outskin_mesh_file = cfns.bet_outskin_mesh_file

    # Get native (mri) voxel index to scaled native (mri) transform
    xform_nativeindex2scalednative = _get_sform(outskin_mesh_file)["trans"]

    # Put MRI-derived headshape points into native space (in mm)
    mri_headshape_nativeindex = _niimask2indexpointcloud(outskin_mesh_file)
    mri_headshape_scalednative = _xform_points(
        xform_nativeindex2scalednative, mri_headshape_nativeindex
    )

    # Put MRI-derived headshape points into HEAD space
    mri_headshape_head = _xform_points(
        xform_scalednative2head, mri_headshape_scalednative
    )

    # ----------------------------------------------------------------------
    # 5) We have the polhemus-derived headshape points in HEAD space and
    #    the MRI-derived headshape (scalp surface) in native MRI space. We
    #    use these to estimate the affine xform from native MRI space using
    #    the ICP algorithm initilaised using the xform estimate in step 2.
    # ----------------------------------------------------------------------

    if use_headshape:
        print("Running ICP...")

        # Run ICP with multiple initialisations to refine registration of
        # MRI-derived headshape points to polhemus derived headshape points,
        # with both in HEAD space

        # Combined polhemus-derived headshape points and polhemus-derived
        # fiducials, with them both in HEAD space. These are the "source"
        # points that will be moved around
        polhemus_headshape_head_4icp = np.concatenate(
            (polhemus_headshape_head, polhemus_fid_head),
            axis=1,
        )

        xform_icp, _, e = _rhino_icp(
            mri_headshape_head,
            polhemus_headshape_head_4icp,
            n_init=n_init,
        )

    else:
        # No refinement by ICP:
        xform_icp = np.eye(4)

    # Create refined xforms using result from ICP
    xform_scalednative2head_refined = np.linalg.inv(xform_icp) @ xform_scalednative2head

    # Put MRI-derived fiducials into refined HEAD space
    mri_nasion_head = _xform_points(
        xform_scalednative2head_refined, mri_nasion_scalednative
    )
    mri_rpa_head = _xform_points(xform_scalednative2head_refined, mri_rpa_scalednative)
    mri_lpa_head = _xform_points(xform_scalednative2head_refined, mri_lpa_scalednative)

    # ---------------
    # Save coreg info
    # ---------------

    # Save xforms in MNE format in mm

    # Save xform from head to mri for the scaled mri
    head_scaledmri_t = Transform(
        "head", "mri", np.linalg.inv(xform_scalednative2head_refined)
    )
    write_trans(cfns.head_scaledmri_t_file, head_scaledmri_t, overwrite=True)

    # Save xform from head to mri for the unscaled mri, this is needed if
    # we later want to map back into MNI space from head space following
    # source recon, i.e. by combining this xform with sfns.mni_mri_t_file
    xform_native2head_refined = (
        np.linalg.inv(xform_icp) @ xform_scalednative2head @ xform_native2scalednative
    )
    xform_native2head_refined_copy = np.copy(xform_native2head_refined)
    head_mri_t = Transform("head", "mri", np.linalg.inv(xform_native2head_refined_copy))
    write_trans(cfns.head_mri_t_file, head_mri_t, overwrite=True)

    # Save xform from mrivoxel to mri
    nativeindex_scalednative_t = np.copy(xform_nativeindex2scalednative)
    mrivoxel_scaledmri_t = Transform("mri_voxel", "mri", nativeindex_scalednative_t)
    write_trans(cfns.mrivoxel_scaledmri_t_file, mrivoxel_scaledmri_t, overwrite=True)

    # Save MRI derived fiducials in mm in HEAD space
    np.savetxt(cfns.mri_nasion_file, mri_nasion_head)
    np.savetxt(cfns.mri_rpa_file, mri_rpa_head)
    np.savetxt(cfns.mri_lpa_file, mri_lpa_head)

    # ------------------------------------------------------------------------
    # Create MRI-derived freesurfer meshes in native/mri space in mm, for use
    # by forward modelling
    # ------------------------------------------------------------------------

    nativeindex_scalednative_t = np.copy(xform_nativeindex2scalednative)
    mrivoxel_scaledmri_t = Transform("mri_voxel", "mri", nativeindex_scalednative_t)
    _create_freesurfer_meshes_from_bet_surfaces(cfns, mrivoxel_scaledmri_t["trans"])

    if use_nose:
        # Not needed by the forward model, but used to report and plot the
        # coregistration
        _create_freesurfer_mesh_from_bet_surface(
            infile=cfns.bet_outskin_plus_nose_mesh_file,
            surf_outfile=cfns.bet_outskin_plus_nose_surf_file,
            nii_mesh_file=cfns.bet_outskin_plus_nose_mesh_file,
            xform_mri_voxel2mri=mrivoxel_scaledmri_t["trans"],
        )

    # ----------------------------------
    # Report and plot the coregistration
    # ----------------------------------

    dist = coreg_error(session, include_nose=use_nose)
    quality = {
        "rms": float(np.sqrt(np.mean(dist**2))),
        "max": float(np.max(np.abs(dist))),
    }
    save_params(session.coreg.coreg_quality_file, quality)
    print(
        f"Headshape points to scalp: rms {quality['rms']:.1f} mm, "
        f"max {quality['max']:.1f} mm"
    )

    if plot_type is not None:
        filename = f"{session.coreg_dir}/coreg.{plot_type}"
        plot_coregistration(
            session, include_nose=use_nose, filename=filename, show=show
        )

    print("Coregistration complete.")


def coreg_error(session: Session, include_nose: bool = False) -> np.ndarray:
    """Distance from each headshape point to the scalp after coregistration.

    The quantitative companion to :func:`plot_coregistration`: how far the
    digitised head points land from the MRI-derived scalp once the head has
    been coregistered to the MRI. Use this when you want the individual points,
    e.g. to find which electrodes are displaced; :func:`coreg_rms` is the usual
    single number summary.

    What the distances mean depends on how the coregistration was run. With
    ``use_headshape=True`` the ICP fits the headshape points to this surface,
    so they are a goodness-of-fit residual. With ``use_headshape=False`` the
    fit is driven by the fiducials alone, so they are an independent check on
    the digitised positions.

    Parameters
    ----------
    session : Session
        Files of the session.
    include_nose : bool, optional
        Should we measure against the outskin surface with the nose? Pass the
        same value used for ``use_nose`` in :func:`coregister_head_and_mri`.

    Returns
    -------
    dist : np.ndarray
        Signed distance in mm from each headshape point to the nearest point
        on the scalp surface. Positive is outside the scalp, negative inside.
    """

    # RHINO does everything in mm

    cfns = session.coreg

    if include_nose:
        outskin_surf_file = cfns.bet_outskin_plus_nose_surf_file
    else:
        outskin_surf_file = cfns.bet_outskin_surf_file

    # Headshape points are in mm in HEAD space (none when the coregistration
    # used only the fiducials)
    if not os.path.exists(cfns.head_headshape_file):
        return {"distances": [], "rms": float("nan"), "mean": float("nan")}
    headshape_head = np.loadtxt(cfns.head_headshape_file)
    head_scaledmri_t = read_trans(cfns.head_scaledmri_t_file)
    headshape_mri = _xform_points(head_scaledmri_t["trans"], headshape_head).T

    vertices, _ = nib.freesurfer.read_geometry(outskin_surf_file)
    dist, index = KDTree(vertices).query(headshape_mri)

    # Sign each distance by whether the point sits further from the centre of
    # the scalp than the scalp surface does
    centre = vertices.mean(axis=0)
    outside = np.linalg.norm(headshape_mri - centre, axis=1) > np.linalg.norm(
        vertices[index] - centre, axis=1
    )
    return np.where(outside, dist, -dist)


def coreg_rms(session: Session, include_nose: bool = False) -> float:
    """How far the headshape points are from the scalp after coregistration.

    The root mean square of :func:`coreg_error`, as a single number per
    session so batches can be ranked and the bad fits dropped. See
    :func:`coreg_error` for what the number means.

    Parameters
    ----------
    session : Session
        Files of the session.
    include_nose : bool, optional
        Should we measure against the outskin surface with the nose? Pass the
        same value used for ``use_nose`` in :func:`coregister_head_and_mri`.

    Returns
    -------
    rms : float
        Root mean square distance in mm.
    """
    dist = coreg_error(session, include_nose=include_nose)
    return float(np.sqrt(np.mean(dist**2)))


def plot_coregistration(
    session: Session,
    display_outskin: bool = True,
    display_sensors: bool = True,
    display_sensor_oris: bool = True,
    display_fiducials: bool = True,
    display_headshape_pnts: bool = True,
    include_nose: bool = True,
    filename: str | None = None,
    show: bool = False,
) -> None:
    """Plot coregistration.

    Parameters
    ----------
    session : Session
        Files of the session.
    display_outskin : bool, optional
        Whether to show scalp surface in the display.
    display_sensors : bool, optional
        Whether to include sensors in the display.
    display_sensor_oris : bool, optional
        Whether to include sensor orientations in the display.
    display_fiducials : bool, optional
        Whether to include fiducials in the display.
    display_headshape_pnts : bool, optional
        Whether to include headshape points in the display.
    include_nose : bool, option
        Should we use the outskin surface with the nose?
    filename : str, optional
        Filename to save display to (as an interactive html).
        Must have extension .html.
    show : bool, optional
        Should we show the plots? Only used if filename has
        extension '.png'.
    """

    # Note the jargon used varies for xforms and coord spaces:
    # MEG (device): dev_head_t --> HEAD (polhemus)
    # HEAD (polhemus): head_mri_t (polhemus2native) --> MRI (native)
    # MRI (native): mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices

    # RHINO does everything in mm

    print("Plotting coregistration")

    if filename is None:
        filename = f"{session.coreg_dir}/coreg.png"

    coreg = session.coreg

    bet_outskin_mesh_file = coreg.bet_outskin_mesh_file
    bet_outskin_mesh_vtk_file = coreg.bet_outskin_mesh_vtk_file
    bet_outskin_surf_file = coreg.bet_outskin_surf_file

    bet_outskin_plus_nose_mesh_file = coreg.bet_outskin_plus_nose_mesh_file
    bet_outskin_plus_nose_surf_file = coreg.bet_outskin_plus_nose_surf_file

    head_scaledmri_t_file = coreg.head_scaledmri_t_file
    mrivoxel_scaledmri_t_file = coreg.mrivoxel_scaledmri_t_file
    mri_nasion_file = coreg.mri_nasion_file
    mri_rpa_file = coreg.mri_rpa_file
    mri_lpa_file = coreg.mri_lpa_file
    head_nasion_file = coreg.head_nasion_file
    head_rpa_file = coreg.head_rpa_file
    head_lpa_file = coreg.head_lpa_file
    head_headshape_file = coreg.head_headshape_file
    info_fif_file = coreg.info_fif_file

    if include_nose:
        outskin_mesh_file = bet_outskin_plus_nose_mesh_file
        outskin_mesh_4surf_file = bet_outskin_plus_nose_mesh_file
        outskin_surf_file = bet_outskin_plus_nose_surf_file
    else:
        outskin_mesh_file = bet_outskin_mesh_file
        outskin_mesh_4surf_file = bet_outskin_mesh_vtk_file
        outskin_surf_file = bet_outskin_surf_file

    # ------------
    # Setup xforms
    # ------------
    info = mne.io.read_info(info_fif_file)

    mrivoxel_scaledmri_t = read_trans(mrivoxel_scaledmri_t_file)
    head_scaledmri_t = read_trans(head_scaledmri_t_file)
    dev_head_t, _ = _get_trans(info["dev_head_t"], "meg", "head")

    # Change xform from metres to mm.
    # Note that MNE xform in fif.info assume metres, whereas we want it in mm.
    # To change units for an xform, just need to change the translation part
    # and leave the rotation alone.
    dev_head_t["trans"][0:3, -1] = dev_head_t["trans"][0:3, -1] * 1000

    # We are going to display everything in MEG (device) coord frame in mm
    head_trans = invert_transform(dev_head_t)
    meg_trans = Transform("meg", "meg")
    mri_trans = invert_transform(
        combine_transforms(dev_head_t, head_scaledmri_t, "meg", "mri")
    )

    # ------------------------------------
    # Setup fiducials and headshape points
    # ------------------------------------
    if display_fiducials:
        # Load polhemus-derived fiducials, these are in mm in HEAD space
        polhemus_nasion_meg = None
        if os.path.isfile(head_nasion_file):
            polhemus_nasion_head = np.loadtxt(head_nasion_file)
            polhemus_nasion_meg = _xform_points(
                head_trans["trans"], polhemus_nasion_head
            )
        polhemus_rpa_meg = None
        if os.path.isfile(head_rpa_file):
            polhemus_rpa_head = np.loadtxt(head_rpa_file)
            polhemus_rpa_meg = _xform_points(head_trans["trans"], polhemus_rpa_head)
        polhemus_lpa_meg = None
        if os.path.isfile(head_lpa_file):
            polhemus_lpa_head = np.loadtxt(head_lpa_file)
            polhemus_lpa_meg = _xform_points(head_trans["trans"], polhemus_lpa_head)

        # Load MRI derived fiducials, these are in mm in HEAD space
        mri_nasion_meg = None
        if os.path.isfile(mri_nasion_file):
            mri_nasion_head = np.loadtxt(mri_nasion_file)
            mri_nasion_meg = _xform_points(head_trans["trans"], mri_nasion_head)
        mri_rpa_meg = None
        if os.path.isfile(mri_rpa_file):
            mri_rpa_head = np.loadtxt(mri_rpa_file)
            mri_rpa_meg = _xform_points(head_trans["trans"], mri_rpa_head)
        mri_lpa_meg = None
        if os.path.isfile(mri_lpa_file):
            mri_lpa_head = np.loadtxt(mri_lpa_file)
            mri_lpa_meg = _xform_points(head_trans["trans"], mri_lpa_head)
    else:
        polhemus_nasion_meg = polhemus_rpa_meg = polhemus_lpa_meg = None
        mri_nasion_meg = mri_rpa_meg = mri_lpa_meg = None

    if display_headshape_pnts:
        polhemus_headshape_meg = None
        if os.path.isfile(head_headshape_file):
            polhemus_headshape_head = np.loadtxt(head_headshape_file)
            polhemus_headshape_meg = _xform_points(
                head_trans["trans"], polhemus_headshape_head
            )
    else:
        polhemus_headshape_meg = None

    # -----------------
    # Setup MEG sensors
    # -----------------
    meg_rrs, meg_tris, meg_sensor_locs, meg_sensor_oris = [], [], [], []
    try:
        if display_sensors or display_sensor_oris:
            meg_picks = mne.pick_types(info, meg=True, ref_meg=False, exclude=())
            coil_transs = [
                mne._fiff.tag._loc_to_coil_trans(info["chs"][pick]["loc"])
                for pick in meg_picks
            ]
            coils = mne.forward._create_meg_coils(
                [info["chs"][pick] for pick in meg_picks], acc="normal"
            )

            degenerate_sensor_indices = []
            offset = 0
            sensor_idx = 0

            for coil, coil_trans in zip(coils, coil_transs):
                try:
                    rrs, tris, *extra_z = mne.viz._3d._sensor_shape(coil)
                    if extra_z:
                        rrs = rrs + [0.0, 0.0, extra_z[0]]
                except Exception as exc:
                    is_qhull = (
                        isinstance(exc, RuntimeError)
                        or "Qhull" in repr(exc)
                        or "Initial simplex is flat" in str(exc)
                    )
                    if is_qhull:
                        # Create a tiny 3-point triangle in metres (so later transforms work)
                        tiny = 0.001  # 1 mm
                        rrs = np.array(
                            [[0.0, 0.0, 0.0], [tiny, 0.0, 0.0], [-tiny, 0.0, 0.0]]
                        )
                        tris = np.array([[0, 1, 2]])
                        degenerate_sensor_indices.append(sensor_idx)
                    else:
                        # Unexpected exception - re-raise
                        raise

                # apply coil transform to get coil shape in device coords (metres)
                rrs = apply_trans(coil_trans, rrs)
                meg_rrs.append(rrs)
                meg_tris.append(tris + offset)

                # sensor location: origin transformed by coil_trans
                sens_locs = np.array([[0.0, 0.0, 0.0]])
                sens_locs = apply_trans(coil_trans, sens_locs)

                # orientation: unit z vector (small scale)
                sens_oris = np.array([[0.0, 0.0, 1.0]]) * 0.01
                sens_oris = apply_trans(coil_trans, sens_oris)
                sens_oris = sens_oris - sens_locs

                meg_sensor_locs.append(sens_locs)
                meg_sensor_oris.append(sens_oris)

                offset += len(rrs)
                sensor_idx += 1

            if len(meg_rrs) == 0:
                print("MEG sensors not found. Cannot plot MEG locations.")
            else:
                meg_rrs = apply_trans(meg_trans, np.concatenate(meg_rrs, axis=0))
                meg_sensor_locs = apply_trans(
                    meg_trans, np.concatenate(meg_sensor_locs, axis=0)
                )
                meg_sensor_oris = apply_trans(
                    meg_trans, np.concatenate(meg_sensor_oris, axis=0)
                )
                meg_tris = np.concatenate(meg_tris, axis=0)

            # convert to mm
            meg_rrs = meg_rrs * 1000
            meg_sensor_locs = meg_sensor_locs * 1000
            meg_sensor_oris = meg_sensor_oris * 1000

    except Exception as e:
        # If anything goes catastrophically wrong in sensor setup, report and continue
        print(f"Warning: problem setting up MEG sensors: {e}")
        meg_rrs = np.array([]).reshape((0, 3))
        meg_tris = np.array([]).reshape((0, 3))
        meg_sensor_locs = np.array([]).reshape((0, 3))
        meg_sensor_oris = np.array([]).reshape((0, 3))

    # --------
    # Do plots
    # --------
    import pyvista

    pyvista.OFF_SCREEN = True
    mne.viz.set_3d_backend("notebook")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Initialize figure
        renderer = _get_renderer(None, bgcolor=(0.5, 0.5, 0.5), size=(500, 500))

        # Headshape points
        if display_headshape_pnts:
            if polhemus_headshape_meg is not None and len(polhemus_headshape_meg.T) > 0:
                polhemus_headshape_megt = polhemus_headshape_meg.T
                if len(polhemus_headshape_megt) < 200:
                    scale = 0.007
                elif (
                    len(polhemus_headshape_megt) >= 200
                    and len(polhemus_headshape_megt) < 400
                ):
                    scale = 0.005
                elif len(polhemus_headshape_megt) >= 400:
                    scale = 0.003
                color, alpha = "red", 1
                renderer.sphere(
                    center=polhemus_headshape_megt,
                    color=color,
                    scale=scale * 1000,
                    opacity=alpha,
                    backface_culling=True,
                )
            else:
                print("There are no headshape points to display")

        # Fiducials
        if display_fiducials:
            # MRI-derived nasion, rpa, lpa
            if mri_nasion_meg is not None and len(mri_nasion_meg.T) > 0:
                color, scale, alpha = "yellow", 0.09, 1
                for data in [mri_nasion_meg.T, mri_rpa_meg.T, mri_lpa_meg.T]:
                    transform = np.eye(4)
                    transform[:3, :3] = mri_trans["trans"][:3, :3] * scale * 1000
                    transform = transform @ rotation(0, 0, np.pi / 4)
                    renderer.quiver3d(
                        x=data[:, 0],
                        y=data[:, 1],
                        z=data[:, 2],
                        u=1.0,
                        v=0.0,
                        w=0.0,
                        color=color,
                        mode="oct",
                        scale=scale,
                        opacity=alpha,
                        backface_culling=True,
                        solid_transform=transform,
                    )
            else:
                print("There are no MRI derived fiducials to display")

            # Polhemus-derived nasion, rpa, lpa
            if polhemus_nasion_meg is not None and len(polhemus_nasion_meg.T) > 0:
                color, scale, alpha = "pink", 0.012, 1
                for data in [
                    polhemus_nasion_meg.T,
                    polhemus_rpa_meg.T,
                    polhemus_lpa_meg.T,
                ]:
                    renderer.sphere(
                        center=data,
                        color=color,
                        scale=scale * 1000,
                        opacity=alpha,
                        backface_culling=True,
                    )
            else:
                print("There are no polhemus derived fiducials to display")

        # Sensors
        if display_sensors:
            if len(meg_rrs) > 0:
                color, alpha = (0.0, 0.25, 0.5), 0.2
                surf = dict(rr=meg_rrs, tris=meg_tris)
                renderer.surface(
                    surface=surf,
                    color=color,
                    opacity=alpha,
                    backface_culling=True,
                )
            else:
                print("No sensor surfaces available to display")

        # Sensor orientations (arrows)
        if display_sensor_oris:
            if len(meg_sensor_locs) > 0:
                color, scale = (0.0, 0.25, 0.5), 15
                renderer.quiver3d(
                    x=meg_sensor_locs[:, 0],
                    y=meg_sensor_locs[:, 1],
                    z=meg_sensor_locs[:, 2],
                    u=meg_sensor_oris[:, 0],
                    v=meg_sensor_oris[:, 1],
                    w=meg_sensor_oris[:, 2],
                    color=color,
                    mode="arrow",
                    scale=scale,
                    backface_culling=False,
                )

        # Outskin surface
        if display_outskin:
            _create_freesurfer_mesh_from_bet_surface(
                infile=outskin_mesh_4surf_file,
                surf_outfile=outskin_surf_file,
                nii_mesh_file=outskin_mesh_file,
                xform_mri_voxel2mri=mrivoxel_scaledmri_t["trans"],
            )
            coords_native, faces = nib.freesurfer.read_geometry(outskin_surf_file)

            coords_meg = _xform_points(mri_trans["trans"], coords_native.T).T
            surf_mri = dict(rr=coords_meg, tris=faces)

            renderer.surface(
                surface=surf_mri,
                color=(0, 0.7, 0.7),
                opacity=0.4,
                backface_culling=False,
            )

        renderer.set_camera(
            azimuth=90,
            elevation=90,
            distance=600,
            focalpoint=(0.0, 0.0, 0.0),
        )

        # Save
        ext = Path(filename).suffix.lower()

        if ext == ".html":
            print(f"Saving {filename}")
            renderer.figure.plotter.export_html(filename)

        elif ext == ".png":
            # Capture three views and composite into a single PNG
            views = [
                (
                    "Frontal",
                    dict(
                        azimuth=90,
                        elevation=90,
                        distance=600,
                        focalpoint=(0.0, 0.0, 0.0),
                    ),
                ),
                (
                    "Right",
                    dict(
                        azimuth=0,
                        elevation=90,
                        distance=600,
                        focalpoint=(0.0, 0.0, 0.0),
                    ),
                ),
                (
                    "Top",
                    dict(
                        azimuth=90,
                        elevation=0,
                        distance=600,
                        focalpoint=(0.0, 0.0, 0.0),
                    ),
                ),
            ]

            plotter = renderer.figure.plotter
            screenshots = []

            for name, cam in views:
                renderer.set_camera(
                    azimuth=cam["azimuth"],
                    elevation=cam["elevation"],
                    distance=cam["distance"],
                    focalpoint=cam["focalpoint"],
                )
                screenshots.append(plotter.screenshot())

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for ax, img, (name, _) in zip(axes, screenshots, views):
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(name, fontsize=22)
            if display_headshape_pnts:
                rms = coreg_rms(session, include_nose=include_nose)
                fig.suptitle(
                    f"headshape to scalp: rms {rms:.1f} mm",
                    fontsize=18,
                    y=1.02,
                )
            fig.tight_layout()
            print(f"Saving {filename}")
            fig.savefig(filename, dpi=150, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)

        else:
            raise ValueError("Extension must be png or html.")


def _rhino_icp(
    mri_headshape_head: np.ndarray,
    polhemus_headshape_head: np.ndarray,
    n_init: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Runs Iterative Closest Point (ICP) with multiple initialisations.

    Parameters
    ----------
    smri_headshape_polhemus : numpy.ndarray
        [3 x N] locations of the headshape points from MRI in HEAD space
    polhemus_headshape_polhemus : numpy.ndarray
        [3 x N] locations of the headshape points from polhemus in HEAD space.
    n_init : int
        Number of random initialisations to perform.

    Returns
    -------
    xform : numpy.ndarray
        [4 x 4] rigid transformation matrix mapping data2 to data.

    Notes
    -----
    Based on Matlab version from Adam Baker 2014.
    """
    data1 = mri_headshape_head
    data2 = polhemus_headshape_head
    err_old = np.inf
    err = np.zeros(n_init)
    Mr = np.eye(4)
    incremental = False
    if incremental:
        Mr_total = np.eye(4)
    data2r = data2
    for init in range(n_init):
        Mi, distances, i = _icp(data2r.T, data1.T)
        e = np.sqrt(np.mean(np.square(distances)))
        err[init] = e
        if err[init] < err_old:
            print(f"ICP found better xform, error={e}")
            err_old = e
            if incremental:
                Mr_total = Mr @ Mr_total
                xform = Mi @ Mr_total
            else:
                xform = Mi @ Mr
        a = (np.random.uniform() - 0.5) * np.pi / 6
        b = (np.random.uniform() - 0.5) * np.pi / 6
        c = (np.random.uniform() - 0.5) * np.pi / 6
        Rx = np.array(
            [(1, 0, 0), (0, np.cos(a), -np.sin(a)), (0, np.sin(a), np.cos(a))]
        )
        Ry = np.array(
            [(np.cos(b), 0, np.sin(b)), (0, 1, 0), (-np.sin(b), 0, np.cos(b))]
        )
        Rz = np.array(
            [(np.cos(c), -np.sin(c), 0), (np.sin(c), np.cos(c), 0), (0, 0, 1)]
        )
        T = 10 * np.array(
            [
                np.random.uniform() - 0.5,
                np.random.uniform() - 0.5,
                np.random.uniform() - 0.5,
            ]
        )
        Mr = np.eye(4)
        Mr[0:3, 0:3] = Rx @ Ry @ Rz
        Mr[0:3, -1] = np.reshape(T, (1, -1))
        if incremental:
            data2r = Mr @ Mr_total @ np.vstack((data2, np.ones((1, data2.shape[1]))))
        else:
            data2r = Mr @ np.vstack((data2, np.ones((1, data2.shape[1]))))
        data2r = data2r[0:3, :]
    return xform, err, err_old


def _icp(
    A: np.ndarray,
    B: np.ndarray,
    init_pose: np.ndarray | None = None,
    max_iterations: int = 50,
    tolerance: float = 0.0001,
) -> tuple[np.ndarray, np.ndarray, int]:
    """The Iterative Closest Point method:
    finds best-fit transform that maps points A on to points B.

    Parameters
    ----------
    A : numpy.ndarray
        Nxm numpy array of source mD points.
    B : numpy.ndarray
        Nxm numpy array of destination mD point.
    init_pose : numpy.ndarray
        (m+1)x(m+1) homogeneous transformation.
    max_iterations : int
        Exit algorithm after max_iterations.
    tolerance : float
        Convergence criteria.

    Returns
    -------
    T : numpy.ndarray
        (4 x 4) Final homogeneous transformation that maps A on to B.
    distances : numpy.ndarray
        Euclidean distances (errors) of the nearest neighbor.
    i : float
        Number of iterations to converge.

    Notes
    -----
    From: https://github.com/ClayFlannigan/icp/blob/master/icp.py
    """
    m = A.shape[1]
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)
    if init_pose is not None:
        src = np.dot(init_pose, src)
    prev_error = 0
    kdtree = KDTree(dst[:m, :].T)
    for i in range(max_iterations):
        distances, indices = kdtree.query(src[:m, :].T)
        T = _best_fit_transform(src[:m, :].T, dst[:m, indices].T)
        src = np.dot(T, src)
        mean_error = np.sqrt(np.mean(np.square(distances)))
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    T = _best_fit_transform(A, src[:m, :].T)
    return T, distances, i


def _best_fit_transform(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Calculates the least-squares best-fit transform that maps corresponding
    points A to B in m spatial dimensions.

    Parameters
    ----------
    A : numpy.ndarray
        Nxm numpy array of corresponding points.
    B : numpy.ndarray
        Nxm numpy array of corresponding points.

    Outputs
    -------
    T : numpy.ndarray
        (m+1)x(m+1) homogeneous transformation matrix that maps A on to B.
    """
    assert A.shape == B.shape
    m = A.shape[1]
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)
    t = centroid_B.T - np.dot(R, centroid_A.T)
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t
    return T


def _rigid_transform_3D(
    B: np.ndarray, A: np.ndarray, compute_scaling: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate affine transform from points in A to point in B.

    Parameters
    ----------
    A : numpy.ndarray
        3 x num_points. Set of points to register from.
    B : numpy.ndarray
        3 x num_points. Set of points to register to.

    compute_scaling : bool
        Do we compute a scaling on top of rotation and translation?

    Returns
    -------
    xform : numpy.ndarray
        Calculated affine transform, does not include scaling.
    scaling_xform : numpy.ndarray
        Calculated scaling transform (a diagonal 4x4 matrix),
        does not include rotation or translation.

    see http://nghiaho.com/?page_id=671
    """
    assert A.shape == B.shape
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)
    Am = A - centroid_A
    Bm = B - centroid_B
    H = Am @ np.transpose(Bm)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    scaling_xform = np.eye(4)
    if compute_scaling:
        RAm = R @ Am
        U2, S2, V2t = np.linalg.svd(Bm @ np.linalg.pinv(RAm))
        S2 = np.identity(3) * np.mean(S2[S2 > 1e-9])
        scaling_xform[0:3, 0:3] = S2
    t = -R @ centroid_A + centroid_B
    xform = np.eye(4)
    xform[0:3, 0:3] = R
    xform[0:3, -1] = np.reshape(t, (1, -1))
    return xform, scaling_xform
