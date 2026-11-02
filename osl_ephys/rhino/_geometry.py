"""Geometry helpers: NIfTI orientation and sform, point and mesh transforms, FLIRT transforms."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import nibabel as nib
import mne
from mne.transforms import Transform, read_trans


def _get_orient(nii_file: str) -> str:
    """Get orientation of nii file."""
    cmd = f"fslorient -getorient {nii_file}"
    return os.popen(cmd).read().strip()


def _get_sform(nii_file: str) -> Transform:
    """Get sform of nii file."""
    sformcode = int(nib.load(nii_file).header["sform_code"])
    if sformcode == 1 or sformcode == 4:
        sform = nib.load(nii_file).header.get_sform()
    else:
        raise ValueError(
            f"sformcode for {nii_file} is {sformcode}, needs to be 1 or 4.\n\n"
            "The sform code indicates how the sform matrix should be "
            "interpreted:\n"
            "  1 = Scanner Anat (native scanner coordinates)\n"
            "  4 = MNI (MNI-152 standard space)\n\n"
            "How to fix this:\n\n"
            "1. Align the structural to the MNI template, which replaces the "
            "header\n"
            "   with a well defined one (sform code 4). This is the usual fix "
            "for a\n"
            "   defaced structural:\n"
            "     rhino.align_smri_to_mni(mri_file, output_file)\n\n"
            "2. If the qform is valid (check with "
            f"'fslorient -getqformcode {nii_file}'),\n"
            "   copy it to the sform:\n"
            f"     fslorient -copyqform2sform {nii_file}\n\n"
            "3. If the sform matrix is correct but only the code is wrong "
            "(check with\n"
            f"   'fslorient -getsform {nii_file}'), set the code directly:\n"
            f"     fslorient -setsformcode 1 {nii_file}\n\n"
            "4. If the orientation is non-standard, reorient to standard "
            "axes:\n"
            f"     fslreorient2std {nii_file} {nii_file}\n\n"
            "5. If both sform and qform are invalid, re-convert the "
            "original DICOMs\n"
            "   with dcm2niix, which correctly populates both.\n\n"
        )
    sform = mne.Transform("mri_voxel", "mri", sform)
    return sform


def _transform_vtk_mesh(
    vtk_mesh_file_in: str,
    nii_mesh_file_in: str,
    out_vtk_file: str,
    nii_mesh_file_out: str,
    xform_file: str | np.ndarray,
) -> None:
    """
    Outputs mesh to out_vtk_file, which is the result of applying the
    transform xform to vtk_mesh_file_in

    nii_mesh_file_in needs to be the corresponding niftii file from bet
    that corresponds to the same mesh as in vtk_mesh_file_in

    nii_mesh_file_out needs to be the corresponding niftii file from bet
    that corresponds to the same mesh as in out_vtk_file
    """
    rrs_in, tris_in = _get_vtk_mesh_native(vtk_mesh_file_in, nii_mesh_file_in)
    xform_flirtcoords2native_out = _get_flirtcoords2native_xform(nii_mesh_file_out)
    if isinstance(xform_file, str):
        xform = read_trans(xform_file)["trans"]
    else:
        xform = xform_file
    overall_xform = np.linalg.inv(xform_flirtcoords2native_out) @ xform
    rrs_out = _xform_points(overall_xform, rrs_in.T).T
    data = pd.read_csv(vtk_mesh_file_in, sep=r"\s+")
    num_rrs = int(data.iloc[3, 1])
    for col_idx in range(3):
        col = data.columns[col_idx]
        data[col] = data[col].astype(object)
        data.iloc[4 : num_rrs + 4, col_idx] = rrs_out[:, col_idx]
    data.to_csv(out_vtk_file, sep=" ", index=False)


def _get_vtk_mesh_native(
    vtk_mesh_file: str, nii_mesh_file: str
) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(vtk_mesh_file, sep=r"\s+")
    num_rrs = int(data.iloc[3, 1])
    rrs_flirtcoords = data.iloc[4 : num_rrs + 4, 0:3].to_numpy().astype(np.float64)
    xform_flirtcoords2nii = _get_flirtcoords2native_xform(nii_mesh_file)
    rrs_nii = _xform_points(xform_flirtcoords2nii, rrs_flirtcoords.T).T
    num_tris = int(data.iloc[num_rrs + 4, 1])
    tris_nii = (
        data.iloc[num_rrs + 5 : num_rrs + 5 + num_tris, 1:4].to_numpy().astype(int)
    )
    return rrs_nii, tris_nii


def _xform_points(xform: np.ndarray, pnts: np.ndarray) -> np.ndarray:
    """Applies homogeneous linear transformation to an array of 3D coordinates.

    Parameters
    ----------
    xform : numpy.ndarray
        4x4 matrix containing the affine transform.
    pnts : numpy.ndarray
        points to transform, should be 3 x num_points.

    Returns
    -------
    newpnts : numpy.ndarray
        pnts following the xform, will be 3 x num_points.
    """
    if len(pnts.shape) == 1:
        pnts = np.reshape(pnts, [-1, 1])
    num_rows, num_cols = pnts.shape
    if num_rows != 3:
        raise Exception(f"pnts is not 3xN, it is {num_rows}x{num_cols}")
    pnts = np.concatenate((pnts, np.ones([1, pnts.shape[1]])), axis=0)
    newpnts = xform @ pnts
    return newpnts[:3]


def _niimask2indexpointcloud(nii_fname: str, volindex: int | None = None) -> np.ndarray:
    """Takes in a nii.gz mask file name (which equals zero for background
    and != zero for the mask) and returns the mask as a 3 x npoints point cloud.

    Parameters
    ----------
    nii_fname : string
        A nii.gz mask file name (with zero for background, and !=0 for the mask).
    volindex : int
        Volume index, used if nii_mask is a 4D file.

    Returns
    -------
    pc : numpy.ndarray
        3 x npoints point cloud as voxel indices.
    """
    vol = nib.load(nii_fname).get_fdata()
    if len(vol.shape) == 4 and volindex is not None:
        vol = vol[:, :, :, volindex]
    if not len(vol.shape) == 3:
        raise ValueError(
            "nii_mask must be a 3D volume, or nii_mask must be a 4D volume "
            "with volindex specifying a volume index"
        )
    return np.asarray(np.where(vol != 0))


def _get_vol_info_from_nii(mri: str) -> dict:
    """Read volume info from an MRI file.

    Parameters
    ----------
    mri : str
        Path to MRI file.

    Returns
    -------
    out : dict
        Dictionary with keys 'mri_width', 'mri_height', 'mri_depth'
        and 'mri_volume_name'.
    """
    dims = nib.load(mri).get_fdata().shape
    return dict(
        mri_width=dims[0],
        mri_height=dims[1],
        mri_depth=dims[2],
        mri_volume_name=mri,
    )


def _get_flirt_xform_between_axes(from_nii: str, target_nii: str) -> np.ndarray:
    """
    Computes flirt xform that moves from_nii to have voxel indices on the same
    axis as the voxel indices for target_nii.

    Note that this is NOT the same as registration, i.e. the images are not aligned.
    In fact the actual coordinates (in mm) are unchanged.

    It is instead about putting from_nii onto the same axes so that the voxel INDICES
    are comparable. This is achieved by using a transform that sets the sform of
    from_nii to be the same as target_nii without changing the actual coordinates
    (in mm).

    A flirt xform acts on flirtcoords (in mm) rather than on voxel indices or
    on the actual coordinates (see _get_flirtcoords2native_xform), so the
    transform needed to do this is:

      from2targetaxes = inv(targetflirtcoords2target) * fromflirtcoords2from

    where
    - fromflirtcoords2from converts from the flirtcoords of from_nii to its
      actual coordinates
    - targetflirtcoords2target is the same for target_nii

    Both images must be RADIOLOGICAL.
    """
    target2targetflirtcoords = np.linalg.inv(_get_flirtcoords2native_xform(target_nii))
    fromflirtcoords2from = _get_flirtcoords2native_xform(from_nii)
    return target2targetflirtcoords @ fromflirtcoords2from


def _get_mne_xform_from_flirt_xform(
    flirt_xform: np.ndarray, nii_mesh_file_in: str, nii_mesh_file_out: str
) -> np.ndarray:
    """
    Returns a mm coordinates to mm coordinates MNE xform that corresponds to
    the passed in flirt xform.

    Note that we need to do this as flirt xforms include an extra xform based
    on the voxel dimensions (see get_flirtcoords2native_xform).
    """
    flirtcoords2native_xform_in = _get_flirtcoords2native_xform(nii_mesh_file_in)
    flirtcoords2native_xform_out = _get_flirtcoords2native_xform(nii_mesh_file_out)
    return (
        flirtcoords2native_xform_out
        @ flirt_xform
        @ np.linalg.inv(flirtcoords2native_xform_in)
    )


def _get_flirtcoords2native_xform(nii_mesh_file: str) -> np.ndarray:
    """
    Returns xform_flirtcoords2native transform that transforms from flirtcoords
    space in mm into native space in mm, where the passed in nii_mesh_file specifies
    the native space

    Note that for some reason flirt outputs transforms of the form:
    flirt_mni2mri = mri2flirtcoords x mni2mri x flirtcoords2mni

    and bet_surf outputs the .vtk file vertex values in the same flirtcoords mm
    coordinate system.

    See the bet_surf manual:
    https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#betsurf

    If the image has radiological ordering (see fslorient) then the mm coordinates
    are the voxel coordinates scaled by the mm voxel sizes.

    i.e. (x_mm = x_dim * x) where x_mm are the flirtcoords coords in mm, x is the
    voxel coordinate and x_dim is the voxel size in mm.
    """
    mri_orient = _get_orient(nii_mesh_file)
    if mri_orient != "RADIOLOGICAL":
        raise ValueError(
            "Orientation of file must be RADIOLOGICAL, please check output of: "
            f"fslorient -getorient {nii_mesh_file}"
        )
    xform_nativevox2native = _get_sform(nii_mesh_file)["trans"]
    dims = np.append(nib.load(nii_mesh_file).header.get_zooms(), 1)
    xform_flirtcoords2nativevox = np.diag(1.0 / dims)
    return xform_nativevox2native @ xform_flirtcoords2nativevox
