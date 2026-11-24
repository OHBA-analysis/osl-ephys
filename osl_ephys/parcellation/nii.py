"""Utility functions to work with parcellation niftii files.

Example code
------------

import os
import os.path as op
from osl_ephys.source_recon import nii

workingdir = '/Users/woolrich/osl/osl/source_recon/parcellation/files/'
parc_name = 'Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm'

os.system('fslmaths /Users/woolrich/Downloads/{} {}'.format(parc_name, workingdir))

tmpdir = op.join(workingdir, 'tmp')
os.mkdir(tmpdir)
parcel3d_fname = op.join(workingdir, parc_name + '.nii.gz')
parcel4d_fname = op.join(workingdir, parc_name + '_4d.nii.gz')
nii.convert_3dparc_to_4d(parcel3d_fname, parcel4d_fname, tmpdir, 100)

mni_file = '/Users/woolrich/osl/osl/source_recon/parcellation/files/MNI152_T1_8mm_brain.nii.gz'
spatial_res = 8 # mm
parcel4d_ds_fname = op.join(workingdir, parc_name + '_4d_ds' + str(spatial_res) + '.nii.gz')
nii.spatially_downsample(parcel4d_fname, parcel4d_ds_fname, mni_file, spatial_res)

os.system('fslmaths /usr/local/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-prob-2mm.nii.gz -thr 50 -bin /Users/woolrich/osl/osl/source_recon/parcellation/files/HarvardOxford-sub-prob-bin-2mm.nii.gz')

file_in = '/Users/woolrich/osl/osl/source_recon/parcellation/files/Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_4d.nii.gz'
file_out = '/Users/woolrich/osl/osl/source_recon/parcellation/files/HarvOxf-sub-Schaefer100-combined-2mm_4d.nii.gz'
file_append = '/Users/woolrich/osl/osl/source_recon/parcellation/files/HarvardOxford-sub-prob-bin-2mm.nii.gz'
parcel_indices = [3,4,5,6,8,9,10,14,15,16,17,18,19,20] # index from 0
nii.append_4d_parcellation(file_in, file_out, file_append, parcel_indices)

parc_name = '/Users/woolrich/osl/osl/source_recon/parcellation/files/HarvOxf-sub-Schaefer100-combined-2mm_4d'
parcel4d_fname = op.join(parc_name + '.nii.gz')
mni_file = '/Users/woolrich/osl/osl/source_recon/parcellation/files/MNI152_T1_8mm_brain.nii.gz'
spatial_res = 8 # mm
parcel4d_ds_fname = op.join(parc_name + '_ds' + str(spatial_res) + '.nii.gz')
nii.spatially_downsample(parcel4d_fname, parcel4d_ds_fname, mni_file, spatial_res)


fslmaths /Users/woolrich/osl/osl/source_recon/parcellation/files/HarvOxf-sub-Schaefer100-combined-2mm_4d.nii.gz -Tmaxn /Users/woolrich/osl/osl/source_recon/parcellation/files/HarvOxf-sub-Schaefer100-combined-2mm.nii.gz
"""

import os
import os.path as op
import nibabel as nib
import numpy as np
from mne.transforms import Transform
from scipy.spatial import KDTree


def convert_4dparc_to_3d(parcel4d_fname, parcel3d_fname):
    """Convert 4D parcellation to 3D.

    Parameters
    ----------
    parcel4d_fname : str
        4D nifii file, where each volume is a parcel
    parcel3d_fname : str
        3D nifii output file with each voxel with a value of 0 if not in a parcel,
        or 1...p...n_parcels if in parcel p
    """
    os.system("fslmaths {} -Tmaxn -add 1 {}".format(parcel4d_fname, parcel3d_fname))


def convert_3dparc_to_4d(parcel3d_fname, parcel4d_fname, tmpdir, n_parcels):
    """Convert 3D parcellation to 4D.

    Parameters
    ----------
    parcel3d_fname : str
        3D nifii volume with each voxel with a value of 0 if not in a parcel,
        or 1...p...n_parcels if in parcel p
    parcel4d_fname : str
        4D nifii output file, where each volume is a parcel
    tmpdir : str
        temp dir to write to. Must exist.
    n_parcels
        Number of parcels
    """
    os.system("rm -f {}".format(parcel4d_fname))

    vol_list_str = " "
    for pp in range(n_parcels):
        print(pp)
        vol_fname = op.join(tmpdir, "parc3d_vol" + str(pp) + ".nii.gz")
        os.system(
            "fslmaths {} -thr {} -uthr {} -min 1 {}".format(
                parcel3d_fname, pp + 0.5, pp + 1.5, vol_fname
            )
        )
        vol_list_str = vol_list_str + "{} ".format(vol_fname)

    os.system("fslmerge -t {} {}".format(parcel4d_fname, vol_list_str))


def spatially_downsample(file_in, file_out, file_ref, spatial_res):
    """Downsample niftii file file_in spatially and writes it to file_out

    Parameters
    ----------
    file_in: str
    file_out: str
    file_ref: str
        reference niftii volume at resolution spatial_res
    spatial_res
        new spatial res in mm

    """
    os.system(
        "flirt -in {} -ref {} -out {} -applyisoxfm {}".format(
            file_in, file_ref, file_out, spatial_res
        )
    )


def append_4d_parcellation(file_in, file_out, file_append, parcel_indices=None):
    """Appends volumes in file_append to file_in.

    Parameters
    ----------
    file_in : str
    file_out : str
    file_append : str
    parcel_indices : np.ndarray
        (n_indices) numpy array containing volume indices (starting from 0) of volumes from file_append to append to file_in
    """
    if parcel_indices is None:
        nparcels = nib.load(file_append).get_fdata().shape[3]
        parcel_indices = np.arange(nparcels)

    vol_list_str = ""
    for pp in parcel_indices:
        print(pp)
        vol_list_str = vol_list_str + "{},".format(pp)

    os.system(
        "fslselectvols -i {} -o {} --vols={}".format(
            file_append, file_out, vol_list_str
        )
    )
    os.system("fslmerge -t {} {} {}".format(file_out, file_in, file_out))


def get_gridstep(coords):
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
    store = []
    for ii in range(coords.shape[0]):
        store.append(np.sqrt(np.sum(np.square(coords[ii, :] - coords[0, :]))))
    store = np.asarray(store)
    gridstep = int(np.round(np.min(store[np.where(store > 0)]) * 1000))
    return gridstep


def get_sform(nii_file):
    """
    sform allows mapping from simple voxel index coordinates (e.g. from 0 to 256) in scanner space to continuous coordinates (in mm)

    sformcode = os.popen('fslorient -getsformcode {}'.format(nii_file)).read().strip()
    """

    sformcode = int(nib.load(nii_file).header["sform_code"])

    if sformcode == 1 or sformcode == 4:
        sform = nib.load(nii_file).header.get_sform()
    else:
        raise ValueError(
            f"sform code for {nii_file} is {sformcode}, and needs to be 4 or 1.\n"
            "If the qform code is 4 or 1, then considering passing use_qform=True.\n"
            "Also see: https://github.com/OHBA-analysis/osl-ephys/blob/main/examples/misc/fix_smri_files.py"
        )

    sform = Transform("mri_voxel", "mri", sform)
    return sform


def get_orient(nii_file):
    cmd = "fslorient -getorient {}".format(nii_file)

    # use os.popen rather than os.system as we want to return a value, note that this will wait until the read() works before continuing.
    # Without the read() the code will continue without waiting for the system call to finish
    orient = os.popen(cmd).read().strip()

    return orient


def niimask2indexpointcloud(nii_fname, volindex=None):
    """Takes in a nii.gz mask file name (which equals zero for background and != zero for the mask) and returns the mask as a 3 x npoints point cloud.

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
        Exception(
            "nii_mask must be a 3D volume, or nii_mask must be a 4D volume with volindex specifying a volume index"
        )

    # Turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
    pc = np.asarray(np.where(vol != 0))

    return pc


def niimask2mmpointcloud(nii_mask, volindex=None):
    """Takes in a nii.gz mask (which equals zero for background and neq zero for the mask) and returns the mask as a 3 x npoints point cloud in native space in mm's.

    Parameters
    ----------
    nii_mask : string
        A nii.gz mask file name or the [x,y,z] volume (with zero for background, and !=0 for the mask).
    volindex : int
        Volume index, used if nii_mask is a 4D file.

    Returns
    -------
    pc : numpy.ndarray
        3 x npoints point cloud as mm in native space (using sform).
    values : numpy.ndarray
         npoints values.
    """

    vol = nib.load(nii_mask).get_fdata()

    if len(vol.shape) == 4 and volindex is not None:
        vol = vol[:, :, :, volindex]

    if not len(vol.shape) == 3:
        Exception(
            "nii_mask must be a 3D volume, or nii_mask must be a 4D volume with volindex specifying a volume index"
        )

    # Turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
    pc_nativeindex = np.asarray(np.where(vol != 0))

    values = np.asarray(vol[vol != 0])

    # Move from native voxel indices to native space coordinates (in mm)
    pc = xform_points(get_sform(nii_mask)["trans"], pc_nativeindex)

    return pc, values


def xform_points(xform, pnts):
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
    newpnts = newpnts[0:3, :]

    return newpnts


def closest_node(node, nodes):
    """Find nearest node in nodes to the passed in node.

    Returns
    -------
    index : int
        Index to the nearest node in nodes.
    distance : float
        Distance.
    """

    if len(nodes) == 1:
        nodes = np.reshape(nodes, [-1, 1])

    kdtree = KDTree(nodes)
    distance, index = kdtree.query(node)

    return index, distance


def get_vol_info_from_nii(mri):
    """Read volume info from an MRI file.

    Parameters
    ----------
    mri : str
        Path to MRI file.

    Returns
    -------
    out : dict
        Dictionary with keys 'mri_width', 'mri_height', 'mri_depth' and 'mri_volume_name'.
    """
    dims = nib.load(mri).get_fdata().shape
    out = dict(
        mri_width=dims[0], mri_height=dims[1], mri_depth=dims[2], mri_volume_name=mri
    )
    return out


def timeseries2nii(
    timeseries, timeseries_coords, reference_mask_fname, out_nii_fname, times=None
):
    """Maps the (ndipoles,tpts) array of timeseries to the grid defined by reference_mask_fname and outputs them as a niftii file.

    Assumes the timeseries' dipoles correspond to those in reference_mask_fname. Both timeseries and reference_mask_fname are often output from rhino._transform_recon_timeseries_v2.

    Parameters
    ----------
    timeseries : (ndipoles, ntpts) numpy.ndarray
        Time courses. Assumes the timeseries' dipoles correspond to those in reference_mask_fname. Typically derives from rhino._transform_recon_timeseries_v2
    timeseries_coords : (ndipoles, 3) numpy.ndarray
        Coords in mm for dipoles corresponding to passed in timeseries
    reference_mask_fname : string
        A nii.gz mask file name (with zero for background, and !=0 for the mask). Assumes the mask was used to set dipoles for timeseries, typically derived from
        rhino._transform_recon_timeseries_v2
    out_nii_fname : string
        output name of niftii file
    times : (ntpts, ) numpy.ndarray
        Times points in seconds. Assume that times are regularly spaced. Used to set nii file up correctly.

    Returns
    -------
    out_nii_fname : string
        Name of output niftii file
    """

    if len(timeseries.shape) == 1:
        timeseries = np.reshape(timeseries, [-1, 1])

    mni_nii_nib = nib.load(reference_mask_fname)
    coords_ind = niimask2indexpointcloud(reference_mask_fname).T
    coords_mni, tmp = niimask2mmpointcloud(reference_mask_fname)

    mni_nii_values = mni_nii_nib.get_fdata()
    mni_nii_values = np.zeros(np.append(mni_nii_values.shape, timeseries.shape[1]))

    kdtree = KDTree(coords_mni.T)
    gridstep = int(get_gridstep(coords_mni.T) / 1000)

    for ind in range(timeseries_coords.shape[1]):
        distance, index = kdtree.query(timeseries_coords[:, ind])
        # Exclude any timeseries_coords that are further than gridstep away
        # from the best matching coords_mni
        if distance < gridstep:
            mni_nii_values[
                coords_ind[ind, 0], coords_ind[ind, 1], coords_ind[ind, 2], :
            ] = timeseries[ind, :]

    # SAVE AS NIFTI
    vol_nii = nib.Nifti1Image(mni_nii_values, mni_nii_nib.affine)

    vol_nii.header.set_xyzt_units(2)  # mm
    if times is not None:
        vol_nii.header["pixdim"][4] = times[1] - times[0]
        vol_nii.header["toffset"] = -0.5
        vol_nii.header.set_xyzt_units(2, 8)  # mm and secs

    nib.save(vol_nii, out_nii_fname)

    return out_nii_fname
