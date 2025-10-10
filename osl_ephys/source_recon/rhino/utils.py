"""RHINO utilities.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Rob Seymour <rob.seymour@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>

import os
import os.path as op
import subprocess
import pickle
from pathlib import Path
from glob import glob
from shutil import copy

import numpy as np
from sklearn.neighbors import NearestNeighbors
import nibabel as nib
import pandas as pd
from scipy import LowLevelCallable
from scipy.ndimage import generic_filter
from scipy.spatial import KDTree

import matplotlib.pyplot as plt

from mne import Transform
from mne.transforms import read_trans
from mne.surface import write_surface
from mne import io
from mne.channels import make_dig_montage

from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer


import logging
logging.getLogger("numba").setLevel(logging.WARNING)

from fsl import wrappers as fsl_wrappers

from osl_ephys.source_recon.beamforming import transform_recon_timeseries
from osl_ephys.utils.logger import log_or_print
from osl_ephys.utils import soft_import

def get_rhino_filenames(subjects_dir, subject):
    """Get paths to all RHINO files.

    Files will be in subjects_dir/subject/rhino.

    Parameters
    ----------
    subjects_dir : string
        Directory containing the subject directories.
    subject : string
        Subject directory name to put the coregistration files in.

    Returns
    -------
    files : dict
        A dict of files generated and used by RHINO. Contains three keys:
        - 'surf': containing surface extraction file paths.
        - 'coreg': containing coregistration file paths.
        - 'fwd_model': containing the forward model file path.
    """

    # Base RHINO directory
    rhino_dir = op.join(subjects_dir, subject, "rhino")
    if " " in rhino_dir:
        raise ValueError("subjects_dir cannot contain spaces.")

    # Surfaces files
    surfaces_dir = op.join(rhino_dir, "surfaces")
    os.makedirs(surfaces_dir, exist_ok=True)
    surf_files = {
        "basedir": surfaces_dir,
        "smri_file": op.join(surfaces_dir, "smri.nii.gz"),
        "mni2mri_flirt_xform_file": op.join(surfaces_dir, "mni2mri_flirt_xform.txt"),
        "mni_mri_t_file": op.join(surfaces_dir, "mni_mri-trans.fif"),
        "bet_outskin_mesh_vtk_file": op.join(surfaces_dir, "outskin_mesh.vtk"),  # BET output
        "bet_inskull_mesh_vtk_file": op.join(surfaces_dir, "inskull_mesh.vtk"),  # BET output
        "bet_outskull_mesh_vtk_file": op.join(surfaces_dir, "outskull_mesh.vtk"),  # BET output
        "bet_outskin_mesh_file": op.join(surfaces_dir, "outskin_mesh.nii.gz"),
        "bet_outskin_plus_nose_mesh_file": op.join(surfaces_dir, "outskin_plus_nose_mesh.nii.gz"),
        "bet_inskull_mesh_file": op.join(surfaces_dir, "inskull_mesh.nii.gz"),
        "bet_outskull_mesh_file": op.join(surfaces_dir, "outskull_mesh.nii.gz"),
        "std_brain": op.join(os.environ["FSLDIR"], "data", "standard", "MNI152_T1_1mm_brain.nii.gz"),
        "std_brain_bigfov": op.join(os.environ["FSLDIR"], "data", "standard", "MNI152_T1_1mm_BigFoV_facemask.nii.gz"),
        "completed": op.join(surfaces_dir, "completed.txt"),
    }

    # Coregistration files
    coreg_dir = op.join(rhino_dir, "coreg")
    os.makedirs(coreg_dir, exist_ok=True)
    coreg_files = {
        "basedir": coreg_dir,
        "info_fif_file": op.join(coreg_dir, "info-raw.fif"),
        "smri_file": op.join(coreg_dir, "scaled_smri.nii.gz"),
        "head_scaledmri_t_file": op.join(coreg_dir, "head_scaledmri-trans.fif"),
        "head_mri_t_file": op.join(coreg_dir, "head_mri-trans.fif"),
        "ctf_head_mri_t_file": op.join(coreg_dir, "ctf_head_mri-trans.fif"),
        "mrivoxel_scaledmri_t_file": op.join(coreg_dir, "mrivoxel_scaledmri_t_file-trans.fif"),
        "mni_nasion_mni_file": op.join(coreg_dir, "mni_nasion.txt"),
        "mni_rpa_mni_file": op.join(coreg_dir, "mni_rpa.txt"),
        "mni_lpa_mni_file": op.join(coreg_dir, "mni_lpa.txt"),
        "smri_nasion_file": op.join(coreg_dir, "smri_nasion.txt"),
        "smri_rpa_file": op.join(coreg_dir, "smri_rpa.txt"),
        "smri_lpa_file": op.join(coreg_dir, "smri_lpa.txt"),
        "polhemus_nasion_file": op.join(coreg_dir, "polhemus_nasion.txt"),
        "polhemus_rpa_file": op.join(coreg_dir, "polhemus_rpa.txt"),
        "polhemus_lpa_file": op.join(coreg_dir, "polhemus_lpa.txt"),
        "polhemus_headshape_file": op.join(coreg_dir, "polhemus_headshape.txt"),
        "coreg_error_file": op.join(coreg_dir, "coreg_err.txt"),
        # BET mesh output in native space
        "bet_outskin_mesh_vtk_file": op.join(coreg_dir, "scaled_outskin_mesh.vtk"),
        "bet_inskull_mesh_vtk_file": op.join(coreg_dir, "scaled_inskull_mesh.vtk"),
        "bet_outskull_mesh_vtk_file": op.join(coreg_dir, "scaled_outskull_mesh.vtk"),
        # Freesurfer mesh in native space
        # - these are the ones shown in coreg_display() if doing surf plot
        # - these are also used by MNE forward modelling
        "bet_outskin_surf_file": op.join(coreg_dir, "scaled_outskin_surf.surf"),
        "bet_inskull_surf_file": op.join(coreg_dir, "scaled_inskull_surf.surf"),
        "bet_outskull_surf_file": op.join(coreg_dir, "scaled_outskull_surf.surf"),
        "bet_outskin_plus_nose_surf_file": op.join(coreg_dir, "scaled_outskin_plus_nose_surf.surf"),
        # BET output surface mask as nii in native space
        "bet_outskin_mesh_file": op.join(coreg_dir, "scaled_outskin_mesh.nii.gz"),
        "bet_outskin_plus_nose_mesh_file": op.join(coreg_dir, "scaled_outskin_plus_nose_mesh.nii.gz"),
        "bet_inskull_mesh_file": op.join(coreg_dir, "scaled_inskull_mesh.nii.gz"),
        "bet_outskull_mesh_file": op.join(coreg_dir, "scaled_outskull_mesh.nii.gz"),
        "std_brain": op.join(os.environ["FSLDIR"], "data", "standard", "MNI152_T1_1mm_brain.nii.gz"),
    }

    # All RHINO files
    files = {"surf": surf_files, "coreg": coreg_files, "fwd_model": op.join(rhino_dir, "model-fwd.fif")}

    return files


def system_call(cmd, verbose=False):
    """ Run a system call.
    
    Parameters
    ----------
    cmd : string
        Command to run.
    verbose : bool
        Print command before running.
    """
    if verbose:
        log_or_print(cmd)
    subprocess.call(cmd, shell=True)


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
        Exception("nii_mask must be a 3D volume, or nii_mask must be a 4D volume with volindex specifying a volume index")

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
        Exception("nii_mask must be a 3D volume, or nii_mask must be a 4D volume with volindex specifying a volume index")

    # Turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
    pc_nativeindex = np.asarray(np.where(vol != 0))

    values = np.asarray(vol[vol != 0])

    # Move from native voxel indices to native space coordinates (in mm)
    pc = xform_points(get_sform(nii_mask)["trans"], pc_nativeindex)

    return pc, values


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
    out = dict(mri_width=dims[0], mri_height=dims[1], mri_depth=dims[2], mri_volume_name=mri)
    return out


def get_sform(nii_file):
    """
    sform allows mapping from simple voxel index cordinates (e.g. from 0 to 256) in scanner space to continuous coordinates (in mm)

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


def check_nii_for_nan(filename):
    img = nib.load(filename)
    data = img.get_fdata()
    return np.isnan(data).any()


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def majority(values_ptr, len_values, result, data):
    """
    def _majority(buffer, required_majority):
       return buffer.sum() >= required_majority

    See: https://ilovesymposia.com/2017/03/12/scipys-new-lowlevelcallable-is-a-game-changer/

    Numba cfunc that takes in:
    a double pointer pointing to the values within the footprint,
    a pointer-sized integer that specifies the number of values in the footprint,
    a double pointer for the result, and
    a void pointer, which could point to additional parameters
    """
    values = carray(values_ptr, (len_values,), dtype=float64)
    required_majority = 14  # in 3D we have 27 voxels in total
    result[0] = values.sum() >= required_majority
    return 1


def binary_majority3d(img):
    """
    Set a pixel to 1 if a required majority (default=14) or more pixels in its 3x3x3 neighborhood are 1, otherwise, set the pixel to 0. img is a 3D binary image
    """

    if img.dtype != "bool":
        raise ValueError("binary_majority3d(img) requires img to be binary")

    if len(img.shape) != 3:
        raise ValueError("binary_majority3d(img) requires img to be 3D")

    imgout = generic_filter(img, LowLevelCallable(majority.ctypes), size=3).astype(int)

    return imgout


def rigid_transform_3D(B, A, compute_scaling=False):
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
        Calculated scaling transform (a diagonal 4x4 matrix), does not include rotation or translation.

    see http://nghiaho.com/?page_id=671
    """

    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    scaling_xform = np.eye(4)

    if compute_scaling:
        # Bm = C @ R @ Am + error
        # where C = c I is a scalar scaling
        RAm = R @ Am
        U2, S2, V2t = np.linalg.svd(Bm @ np.linalg.pinv(RAm))

        # take average of eigenvalues, accounting for the rank
        S2 = np.identity(3) * np.mean(S2[S2 > 1e-9])

        scaling_xform[0:3, 0:3] = S2

    t = -R @ centroid_A + centroid_B

    xform = np.eye(4)
    xform[0:3, 0:3] = R
    xform[0:3, -1] = np.reshape(t, (1, -1))

    return xform, scaling_xform


def xform_points(xform, pnts):
    """Applies homogenous linear transformation to an array of 3D coordinates.

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


def best_fit_transform(A, B):
    """Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions.

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

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T


def nearest_neighbor(src, dst):
    """Find the nearest (Euclidean) neighbor in dst for each point in src.

    Parameters
    ----------
    src : numpy.ndarray
        Nxm array of points.
    dst : numpy.ndarray
        Nxm array of points.

    Returns
    -------
    distances : numpy.ndarray
        Euclidean distances of the nearest neighbor.
    indices : numpy.ndarray
        dst indices of the nearest neighbor.
    """

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=50, tolerance=0.0001):
    """The Iterative Closest Point method: finds best-fit transform that maps points A on to points B.

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

    # Get number of dimensions
    m = A.shape[1]

    # Make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # Apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    kdtree = KDTree(dst[:m, :].T)

    for i in range(max_iterations):
        # Find the nearest neighbors between the current source and destination points
        #distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        distances, indices = kdtree.query(src[:m, :].T)

        # Compute the transformation between the current source and nearest # destination points
        T = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # Update the current source
        src = np.dot(T, src)

        # Check RMS error
        mean_error = np.sqrt(np.mean(np.square(distances)))

        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Calculate final transformation
    T = best_fit_transform(A, src[:m, :].T)

    return T, distances, i


def rhino_icp(smri_headshape_polhemus, polhemus_headshape_polhemus, n_init=10):
    """Runs Iterative Closest Point (ICP) with multiple initialisations.

    Parameters
    ----------
    smri_headshape_polhemus : numpy.ndarray
        [3 x N] locations of the Headshape points in polehumus space (i.e. MRI scalp surface).
    polhemus_headshape_polhemus : numpy.ndarray
        [3 x N] locations of the Polhemus headshape points in polhemus space.
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

    # These are the "destination" points that are held static
    data1 = smri_headshape_polhemus

    # These are the "source" points that will be moved around
    data2 = polhemus_headshape_polhemus

    err_old = np.inf
    err = np.zeros(n_init)

    Mr = np.eye(4)

    incremental = False
    if incremental:
        Mr_total = np.eye(4)

    data2r = data2

    for init in range(n_init):

        Mi, distances, i = icp(data2r.T, data1.T)

        # RMS error
        e = np.sqrt(np.mean(np.square(distances)))
        err[init] = e

        if err[init] < err_old:

            log_or_print("ICP found better xform, error={}".format(e))

            err_old = e

            if incremental:
                Mr_total = Mr @ Mr_total
                xform = Mi @ Mr_total
            else:
                xform = Mi @ Mr

            if False:
                import matplotlib.pyplot as plt

                # after ICP
                data2r_icp = xform_points(xform, data2)
                plt.figure(frameon=False)
                ax = plt.axes(projection="3d")
                ax.scatter(data1[0, 0:-1:10], data1[1, 0:-1:10], data1[2, 0:-1:10], c="blue", marker=".", s=1)
                ax.scatter(data2[0, :], data2[1, :], data2[2, :], c="red", marker="o", s=5)
                ax.scatter(data2r[0, :], data2r[1, :], data2r[2, :], c="green", marker="o", s=5)
                ax.scatter(data2r_icp[0, :], data2r_icp[1, :], data2r_icp[2, :], c="yellow", marker="o", s=5)
                plt.show()
                plt.draw()

        # Give the registration a kick...
        a = (np.random.uniform() - 0.5) * np.pi / 6
        b = (np.random.uniform() - 0.5) * np.pi / 6
        c = (np.random.uniform() - 0.5) * np.pi / 6

        Rx = np.array([(1, 0, 0), (0, np.cos(a), -np.sin(a)), (0, np.sin(a), np.cos(a))])
        Ry = np.array([(np.cos(b), 0, np.sin(b)), (0, 1, 0), (-np.sin(b), 0, np.cos(b))])
        Rz = np.array([(np.cos(c), -np.sin(c), 0), (np.sin(c), np.cos(c), 0), (0, 0, 1)])

        T = 10 * np.array([np.random.uniform() - 0.5, np.random.uniform() - 0.5, np.random.uniform() - 0.5])
        Mr = np.eye(4)
        Mr[0:3, 0:3] = Rx @ Ry @ Rz
        Mr[0:3, -1] = np.reshape(T, (1, -1))

        if incremental:
            data2r = Mr @ Mr_total @ np.vstack((data2, np.ones((1, data2.shape[1]))))
        else:
            data2r = Mr @ np.vstack((data2, np.ones((1, data2.shape[1]))))

        data2r = data2r[0:3, :]

    return xform, err, err_old


def get_vtk_mesh_native(vtk_mesh_file, nii_mesh_file):
    """
    Returns mesh rrs in native space in mm and the mesh tris for the passed in vtk_mesh_file

    nii_mesh_file needs to be the corresponding niftii file from bet that corresponds to the same mesh as in vtk_mesh_file
    """

    data = pd.read_csv(vtk_mesh_file, sep=r"\s+")

    num_rrs = int(data.iloc[3, 1])

    # these will be in voxel index space
    rrs_flirtcoords = data.iloc[4 : num_rrs + 4, 0:3].to_numpy().astype(np.float64)

    # move from flirtcoords mm to mri mm (native) space
    xform_flirtcoords2nii = get_flirtcoords2native_xform(nii_mesh_file)
    rrs_nii = xform_points(xform_flirtcoords2nii, rrs_flirtcoords.T).T

    num_tris = int(data.iloc[num_rrs + 4, 1])
    tris_nii = data.iloc[num_rrs + 5 : num_rrs + 5 + num_tris, 1:4].to_numpy().astype(int)

    return rrs_nii, tris_nii


def get_flirtcoords2native_xform(nii_mesh_file):
    """
    Returns xform_flirtcoords2native transform that transforms from flirtcoords space in mm into native space in mm, where the passed in nii_mesh_file specifies the native space

    Note that for some reason flirt outputs transforms of the form: flirt_mni2mri = mri2flirtcoords x mni2mri x flirtcoords2mni

    and bet_surf outputs the .vtk file vertex values in the same flirtcoords mm coordinate system.

    See the bet_surf manual: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide#betsurf

    If the image has radiological ordering (see fslorient) then the mm co-ordinates are the voxel co-ordinates scaled by the mm voxel sizes.

    i.e. (x_mm = x_dim * x) where x_mm are the flirtcoords coords in mm, x is the voxel co-ordinate and x_dim is the voxel size in mm.
    """

    # We will assume orientation of the smri is RADIOLOGICAL as RHINO will have made the smri the same orientation as the standard brain nii.
    # But let's just double check that is the case:
    smri_orient = get_orient(nii_mesh_file)
    if smri_orient != "RADIOLOGICAL":
        raise ValueError("Orientation of file must be RADIOLOGICAL, please check output of: fslorient -getorient {}".format(nii_mesh_file))

    xform_nativevox2native = get_sform(nii_mesh_file)["trans"]
    dims = np.append(nib.load(nii_mesh_file).header.get_zooms(), 1)

    # Then calc xform based on x_mm = x_dim * x (see above)
    xform_flirtcoords2nativevox = np.diag(1.0 / dims)
    xform_flirtcoords2native = xform_nativevox2native @ xform_flirtcoords2nativevox

    return xform_flirtcoords2native


def transform_vtk_mesh(vtk_mesh_file_in, nii_mesh_file_in, out_vtk_file, nii_mesh_file_out, xform_file):
    """
    Outputs mesh to out_vtk_file, which is the result of applying the transform xform to vtk_mesh_file_in

    nii_mesh_file_in needs to be the corresponding niftii file from bet that corresponds to the same mesh as in vtk_mesh_file_in

    nii_mesh_file_out needs to be the corresponding niftii file from bet that corresponds to the same mesh as in out_vtk_file
    """

    rrs_in, tris_in = get_vtk_mesh_native(vtk_mesh_file_in, nii_mesh_file_in)

    xform_flirtcoords2native_out = get_flirtcoords2native_xform(nii_mesh_file_out)

    if isinstance(xform_file, str):
        xform = read_trans(xform_file)["trans"]
    else:
        xform = xform_file

    overall_xform = np.linalg.inv(xform_flirtcoords2native_out) @ xform

    # rrs_in are in native nii_in space in mm
    # transform them using the passed in xform
    rrs_out = xform_points(overall_xform, rrs_in.T).T

    data = pd.read_csv(vtk_mesh_file_in, sep=r"\s+")

    num_rrs = int(data.iloc[3, 1])
    data.iloc[4 : num_rrs + 4, 0:3] = rrs_out

    # write new vtk file
    data.to_csv(out_vtk_file, sep=" ", index=False)


def get_mne_xform_from_flirt_xform(flirt_xform, nii_mesh_file_in, nii_mesh_file_out):
    """
    Returns a mm coordinates to mm coordinates MNE xform that corresponds to the passed in flirt xform.

    Note that we need to do this as flirt xforms include an extra xform based on the voxel dimensions (see get_flirtcoords2native_xform).
    """

    flirtcoords2native_xform_in = get_flirtcoords2native_xform(nii_mesh_file_in)
    flirtcoords2native_xform_out = get_flirtcoords2native_xform(nii_mesh_file_out)

    xform = flirtcoords2native_xform_out @ flirt_xform @ np.linalg.inv(flirtcoords2native_xform_in)

    return xform


def get_flirt_xform_between_axes(from_nii, target_nii):
    """
    Computes flirt xform that moves from_nii to have voxel indices on the same axis as  the voxel indices for target_nii.

    Note that this is NOT the same as registration, i.e. the images are not aligned. In fact the actual coordinates (in mm) are unchanged.
    It is instead about putting from_nii onto the same axes so that the voxel INDICES are comparable. This is achieved by using a transform
    that sets the sform of from_nii to be the same as target_nii without changing the actual coordinates (in mm). Transform needed to do this is:

      from2targetaxes = inv(targetvox2target) * fromvox2from

    In more detail, we need the sform for the transformed from_nii to be the same as the sform for the target_nii, without changing the actual coordinates (in mm).
    In other words, we need:

        fromvox2from * from_nii_vox = targetvox2target * from_nii_target_vox

    where

    - fromvox2from is sform for from_nii (i.e. converts from voxel indices to voxel coords in mm)
    - targetvox2target is sform for target_nii
    - from_nii_vox are the voxel indices for from_nii
    - from_nii_target_vox are the voxel indices for from_nii when transformed onto the target axis.

    => from_nii_target_vox = from2targetaxes * from_nii_vox

    where

    - from2targetaxes = inv(targetvox2target) * fromvox2from
    """

    to2tovox = np.linalg.inv(get_sform(target_nii)["trans"])
    fromvox2from = get_sform(from_nii)["trans"]

    from2to = to2tovox @ fromvox2from

    return from2to


def timeseries2nii(timeseries, timeseries_coords, reference_mask_fname, out_nii_fname, times=None):
    """Maps the (ndipoles,tpts) array of timeseries to the grid defined by reference_mask_fname and outputs them as a niftii file.

    Assumes the timeseries' dipoles correspond to those in reference_mask_fname. Both timeseries and reference_mask_fname are often output from rhino.transform_recon_timeseries.

    Parameters
    ----------
    timeseries : (ndipoles, ntpts) numpy.ndarray
        Time courses. Assumes the timeseries' dipoles correspond to those in reference_mask_fname. Typically derives from rhino.transform_recon_timeseries
    timeseries_coords : (ndipoles, 3) numpy.ndarray
        Coords in mm for dipoles corresponding to passed in timeseries
    reference_mask_fname : string
        A nii.gz mask file name (with zero for background, and !=0 for the mask). Assumes the mask was used to set dipoles for timeseries, typically derived from
        rhino.transform_recon_timeseries
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
            mni_nii_values[coords_ind[ind, 0], coords_ind[ind, 1], coords_ind[ind, 2], :] = timeseries[ind, :]

    # SAVE AS NIFTI
    vol_nii = nib.Nifti1Image(mni_nii_values, mni_nii_nib.affine)

    vol_nii.header.set_xyzt_units(2)  # mm
    if times is not None:
        vol_nii.header["pixdim"][4] = times[1] - times[0]
        vol_nii.header["toffset"] = -0.5
        vol_nii.header.set_xyzt_units(2, 8)  # mm and secs

    nib.save(vol_nii, out_nii_fname)

    return out_nii_fname


def recon_timeseries2niftii(
    subjects_dir,
    subject,
    recon_timeseries,
    out_nii_fname,
    spatial_resolution=None,
    reference_brain="mni",
    times=None,
):
    """Converts a (ndipoles,tpts) array of reconstructed timeseries (in head/polhemus space) to the corresponding dipoles in a standard brain grid in MNI space
    and outputs them as a niftii file.

    Parameters
    ----------
    subjects_dir : string
        Directory to find RHINO subject directories in.
    subject : string
        Subject name directory to find RHINO files in.
    recon_timeseries : (ndipoles, ntpts) np.ndarray
        Reconstructed time courses (in head (polhemus) space). Assumes that the dipoles are the same (and in the same order) as those in the forward model,
        rhino_files['fwd_model']. Typically derive from the VolSourceEstimate's output by MNE source recon methods, e.g. mne.beamformer.apply_lcmv,
        obtained using a forward model generated by RHINO.
    spatial_resolution : int
        Resolution to use for the reference brain in mm (must be an integer, or will be cast to nearest int). If None, then the gridstep used in rhino_files['fwd_model'] is used.
    reference_brain : string, 'mni' or 'mri'
        'mni' indicates that the reference_brain is the stdbrain in MNI space.
        'mri' indicates that the reference_brain is the sMRI in native/mri space.
    times : (ntpts, ) np.ndarray
        Times points in seconds. Will assume that these are regularly spaced.

    Returns
    -------
    out_nii_fname : string
        Name of output niftii file.
    reference_brain_fname : string
        Niftii file name of standard brain mask in MNI space at requested resolution, int(stdbrain_resolution) (with zero for background, and !=0 for the mask).
    """

    if len(recon_timeseries.shape) == 1:
        recon_timeseries = np.reshape(recon_timeseries, [recon_timeseries.shape[0], 1])

    # ------------------------------------------------------------------------------------------------
    # Convert the recon_timeseries to the standard space brain dipole grid at the specified resolution

    recon_ts_out, reference_brain_fname, recon_coords_out, recon_indices = transform_recon_timeseries(
        subjects_dir, subject, recon_timeseries=recon_timeseries, spatial_resolution=spatial_resolution, reference_brain=reference_brain,
    )

    # ----------------------------------
    # Output recon_ts_out as niftii file

    out_nii_fname = timeseries2nii(recon_ts_out, recon_coords_out, reference_brain_fname, out_nii_fname, times=times)

    return out_nii_fname, reference_brain_fname


def save_or_show_renderer(renderer, filename):
    """Save or show a renderer.

    Parameters
    ----------
    renderer : mne.viz.backends._notebook._Renderer Object
        MNE renderer object.
    filename : str
        Filename to save display to (as an interactive html). Must have extension .html. If None we display the renderer.
    """
    if filename is None:
        renderer.show()
    else:
        allowed_extensions = [".html", ".pdf", ".svg", ".eps", ".ps", ".tex"]
        ext = Path(filename).suffix
        if ext not in allowed_extensions:
            raise ValueError(f"{ext} not allowed, please use one of the following: {allowed_extensions}")

        log_or_print(f"saving {filename}")
        if ext == ".html":
            logging.getLogger("param.Row").setLevel(logging.ERROR)  # suppress warnings
            logging.getLogger("param.VTKRenderWindowSynchronized").setLevel(logging.ERROR)
            renderer.figure.plotter.export_html(filename)
        elif ext in allowed_extensions:
            renderer.figure.plotter.save_graphic(filename)


def create_freesurfer_meshes_from_bet_surfaces(filenames, xform_mri_voxel2mri):

    # Create sMRI-derived freesurfer surfaces in native/mri space in mm, for use by forward modelling

    create_freesurfer_mesh_from_bet_surface(
        infile=filenames["bet_inskull_mesh_vtk_file"],
        surf_outfile=filenames["bet_inskull_surf_file"],
        nii_mesh_file=filenames["bet_inskull_mesh_file"],
        xform_mri_voxel2mri=xform_mri_voxel2mri,
    )

    create_freesurfer_mesh_from_bet_surface(
        infile=filenames["bet_outskull_mesh_vtk_file"],
        surf_outfile=filenames["bet_outskull_surf_file"],
        nii_mesh_file=filenames["bet_outskull_mesh_file"],
        xform_mri_voxel2mri=xform_mri_voxel2mri,
    )

    create_freesurfer_mesh_from_bet_surface(
        infile=filenames["bet_outskin_mesh_vtk_file"],
        surf_outfile=filenames["bet_outskin_surf_file"],
        nii_mesh_file=filenames["bet_outskin_mesh_file"],
        xform_mri_voxel2mri=xform_mri_voxel2mri,
    )


def create_freesurfer_mesh_from_bet_surface(infile, surf_outfile, xform_mri_voxel2mri, nii_mesh_file=None):
    """Creates surface mesh in .surf format and in native mri space in mm from infile.

    Parameters
    ----------
    infile : string
        Either:
            1) .nii.gz file containing zero's for background and one's for surface
            2) .vtk file generated by bet_surf (in which case the path to the
            structural MRI, smri_file, must be included as an input)
    surf_outfile : string
        Path to the .surf file generated, containing the surface mesh in mm
    xform_mri_voxel2mri : numpy.ndarray
        4x4 array. Transform from voxel indices to native/mri mm.
    nii_mesh_file : string
        Path to the niftii mesh file that is the niftii equivalent of vtk file passed in as infile (only needed if infile is a vtk file).
    """

    pth, name = op.split(infile)
    name, ext = op.splitext(name)

    if ext == ".gz":
        print("Creating surface mesh for {} .....".format(infile))

        # Soft import raising an informative warning if not installed
        o3d = soft_import("open3d")

        name, ext = op.splitext(name)
        if ext != ".nii":
            raise ValueError("Invalid infile. Needs to be a .nii.gz or .vtk file")

        # convert to point cloud in voxel indices
        nii_nativeindex = niimask2indexpointcloud(infile)

        step = 1
        nii_native = xform_points(xform_mri_voxel2mri, nii_nativeindex[:, 0:-1:step])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(nii_native.T)
        pcd.estimate_normals()
        # to obtain a consistent normal orientation
        pcd.orient_normals_towards_camera_location(pcd.get_center())

        # or you might want to flip the normals to make them point outward, not mandatory
        pcd.normals = o3d.utility.Vector3dVector(-np.asarray(pcd.normals))

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)[0]

        #mesh = mesh.simplify_quadric_decimation(nii_nativeindex.shape[1])

        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles).astype(int)

        # output in freesurfer file format
        write_surface(surf_outfile, verts, tris, file_format="freesurfer", overwrite=True)

    elif ext == ".vtk":
        if nii_mesh_file is None:
            raise ValueError("You must specify a nii_mesh_file (niftii format), if infile format is vtk")

        rrs_native, tris_native = get_vtk_mesh_native(infile, nii_mesh_file)

        write_surface(surf_outfile, rrs_native, tris_native, file_format="freesurfer", overwrite=True)

    else:
        raise ValueError("Invalid infile. Needs to be a .nii.gz or .vtk file")


def transform_bet_surfaces(flirt_xform_file, mne_xform_file, filenames, smri_file):
    """Transforms bet surfaces into mne space.
    
    Parameters
    ----------
    flirt_xform_file : string
        Path to flirt xform file.
    mne_xform_file : string
        Path to mne xform file.
    filenames : dict
        Dictionary containing filenames.
    smri_file : string
        Path to structural MRI file.
    """
    
    # Transform betsurf mask/mesh using passed in flirt transform and mne transform
    for mesh_name in {"outskin_mesh", "inskull_mesh", "outskull_mesh"}:
        # xform mask
        # Command: flirt -in <flirt_mesh_file> -ref <smri_file> -interp nearestneighbour -applyxfm -init <flirt_xform_file> -out <out_file>
        fsl_wrappers.flirt(
            op.join(filenames["basedir"], "flirt_" + mesh_name + ".nii.gz"),
            smri_file,
            interp="nearestneighbour",
            applyxfm=True,
            init=flirt_xform_file,
            out=op.join(filenames["basedir"], mesh_name),
        )

        # xform vtk mesh
        transform_vtk_mesh(
            op.join(filenames["basedir"], "flirt_" + mesh_name + ".vtk"),
            op.join(filenames["basedir"], "flirt_" + mesh_name + ".nii.gz"),
            op.join(filenames["basedir"], mesh_name + ".vtk"),
            op.join(filenames["basedir"], mesh_name + ".nii.gz"),
            mne_xform_file,
        )


def extract_rhino_files(old_subjects_dir, new_subjects_dir, subjects="all", exclude=None, gen_report=True):
    """Extract RHINO files.

    This function extracts surfaces and coregistration files calculated in a previous run.

    Parameters
    ----------
    old_subjects_dir : str
        Subjects directory created with an older version of osl-ephys.
    new_subjects_dir : str
        New directory to create.
    subjects : str or list
        Subjects to include. Defaults to 'all'.
    exclude : str or list
        Subjects to exclude.
    gen_report bool
        Should we generate a report?
    """
    # Avoid circular imports
    from osl_ephys.source_recon.rhino import plot_surfaces, coreg_display
    from osl_ephys.report import src_report

    # Validation
    if exclude is None:
        exclude = []

    if isinstance(exclude, str):
        exclude = [exclude]

    if isinstance(subjects, str):
        if subjects != "all":
            subjects = [subjects]

    # Get subjects to extract RHINO files from
    subjects_to_copy = []
    for subject_dir in sorted(glob(f"{old_subjects_dir}/*")):
        subject_name = Path(subject_dir).name
        if (subject_name in ["report", "logs"] + exclude) or (subjects != "all" and subject_name not in subjects):
            continue
        subjects_to_copy.append(subject_name)

    # --------
    # Surfaces

    files = [
        "completed.txt",
        "inskull_mesh.vtk",
        "mni_mri-trans.fif",
        "outskin_mesh.vtk",
        "outskull_mesh.nii.gz",
        "smri.nii.gz",
        "inskull_mesh.nii.gz",
        "mni2mri_flirt_xform.txt",
        "outskin_mesh.nii.gz",
        "outskin_plus_nose_mesh.nii.gz",
        "outskull_mesh.vtk",
    ]

    for subject in subjects_to_copy:
        old_dir = f"{old_subjects_dir}/{subject}/rhino/surfaces"
        new_dir = f"{new_subjects_dir}/{subject}/rhino/surfaces"
        os.makedirs(new_dir, exist_ok=True)

        # Look for each file
        for file in files:
            old_file = f"{old_dir}/{file}"
            new_file = f"{new_dir}/{file}"

            # Copy file if it exists
            if op.exists(old_file):
                copy(old_file, new_file)

            # Special case (okay if missing)
            elif file == "mni2mri_flirt_xform.txt":
                old_file = old_file.replace("xform", "xform_file")
                if op.exists(old_file):
                    copy(old_file, new_file)

            # File is missing
            else:
                raise FileNotFoundError(old_file)

        # Data from old report
        old_report_data = pickle.load(open(f"{old_subjects_dir}/{subject}/report_data.pkl", "rb"))

        # Create surfaces plots
        include_nose = old_report_data["include_nose"]
        surface_plots = plot_surfaces(new_subjects_dir, subject, include_nose)
        surface_plots = [s.replace(f"{new_subjects_dir}/", "") for s in surface_plots]

        # Save info for report
        src_report.add_to_data(
            f"{new_subjects_dir}/{subject}/report_data.pkl",
            {
                "compute_surfaces": True,
                "include_nose": include_nose,
                "do_mri2mniaxes_xform": old_report_data.pop("do_mri2mniaxes_xform", None),
                "surface_plots": surface_plots,
            },
        )

    # --------------
    # Coregistration

    files = [
        "info-raw.fif",
        "polhemus_lpa.txt",
        "polhemus_rpa.txt",
        "polhemus_nasion.txt",
        "polhemus_headshape.txt",
        "smri_lpa.txt",
        "smri_rpa.txt",
        "smri_nasion.txt",
        "head_mri-trans.fif",
        "head_scaledmri-trans.fif",
        "mrivoxel_scaledmri_t_file-trans.fif",
        "scaled_outskin_plus_nose_mesh.nii.gz",
        "scaled_outskin_mesh.nii.gz",
        "scaled_outskull_mesh.nii.gz",
        "scaled_inskull_mesh.nii.gz",
        "scaled_inskull_surf.surf",
        "scaled_outskin_surf.surf",
        "scaled_outskull_surf.surf",
        "scaled_outskin_mesh.vtk",
        "scaled_outskull_mesh.vtk",
        "scaled_inskull_mesh.vtk",
        "scaled_smri.nii.gz",
    ]
    optional_files = [
        "mni_nasion.txt",
        "mni_lpa.txt",
        "mni_rpa.txt",
    ]

    for subject in subjects_to_copy:
        old_dir = f"{old_subjects_dir}/{subject}/rhino/coreg"
        new_dir = f"{new_subjects_dir}/{subject}/rhino/coreg"
        os.makedirs(new_dir, exist_ok=True)

        # Copy each file if it exists
        for file in files:
            old_file = f"{old_dir}/{file}"
            new_file = f"{new_dir}/{file}"
            if op.exists(old_file):
                copy(old_file, new_file)
            else:
                raise FileNotFoundError(old_file)
        for file in optional_files:
            old_file = f"{old_dir}/{file}"
            new_file = f"{new_dir}/{file}"
            if op.exists(old_file):
                copy(old_file, new_file)

        # Special case
        std_brains = glob(f"{old_dir}/MNI152_T1_*_brain.nii.gz")
        for std_brain in std_brains:
            copy(std_brain, std_brain.replace(old_dir, new_dir))

        # Save plot
        coreg_display(
            subjects_dir=new_subjects_dir,
            subject=subject,
            display_outskin_with_nose=False,
            filename=f"{new_dir}/coreg.html",
        )
        coreg_filename = f"{new_dir}/coreg.html".replace(f"{new_subjects_dir}/", "")

        # Data from old report
        old_report_data = pickle.load(open(f"{old_subjects_dir}/{subject}/report_data.pkl", "rb"))

        # Save info for the report
        src_report.add_to_data(
            f"{new_subjects_dir}/{subject}/report_data.pkl",
            {
                "coregister": True,
                "use_headshape": old_report_data["use_headshape"],
                "use_nose": old_report_data["use_nose"],
                "already_coregistered": old_report_data.pop("already_coregistered", None),
                "allow_smri_scaling": old_report_data.pop("allow_smri_scaling", None),
                "n_init_coreg": old_report_data.pop("n_init", None),
                "fid_err": old_report_data.pop("fid_err", None),
                "coreg_plot": coreg_filename,
            },
        )

    # ------
    # Report

    if gen_report:
        # Generate report data for each subject
        reportdir = f"{new_subjects_dir}/report"
        for subject in subjects_to_copy:
            src_report.gen_html_data({"source_recon": [{"Extracted RHINO Files From": f" {old_subjects_dir} "}]}, new_subjects_dir, subject, reportdir)

        # Generate subjects report
        src_report.gen_html_page(reportdir)

        # Generate summary report
        if src_report.gen_html_summary(reportdir):
            log_or_print("******************************" + "*" * len(str(reportdir)))
            log_or_print(f"* REMEMBER TO CHECK REPORT: {reportdir} *")
            log_or_print("******************************" + "*" * len(str(reportdir)))

def save_polhemus_fif(raw, subjects_dir,subject):
    """Save the polhemus information (nasion, LPA, RPA, headshape) from a raw MNE file

    Files will be in subjects_dir/subject/coreg.

    Parameters
    ----------
    raw : raw MNE data structure
        Data loaded into MNE with preload=True
    subjects_dir : string
        Directory containing the subject directories.
    subject : string
        Subject directory name to put the coregistration files in.
    """
    # Check the input is a raw data structure
    if not isinstance(raw, io.BaseRaw):
        log_or_print("The object is not a raw data structure.")
        return

    # Initialize empty lists to store the 3D coordinates
    nasion      = np.empty((0, 3))
    LPA         = np.empty((0, 3))
    RPA         = np.empty((0, 3))
    headshape   = np.empty((0, 3))  

    # Check if there is digitisation information
    if not raw.info["dig"]:
         log_or_print("Error: raw.info['dig'] is empty. Exiting function.")
         return 

    else:
    # Iterate through the digitalization points - the *1000 is for m to mm conversion
        for p in raw.info["dig"]:
            if p["kind"] == 1 and p["ident"] == 1:
                LPA = np.append(LPA,(p["r"]*1000))  # Append the 3D coordinates of LPA
            elif p["kind"] == 1 and p["ident"] == 2:
                nasion = np.append(nasion,p["r"]*1000) # Append the 3D coordinates of nasion
            elif p["kind"] == 1 and p["ident"] == 3:
                RPA = np.append(RPA,(p["r"]*1000))  # Append the 3D coordinates of RPA
            elif p["kind"] == 4:
                headshape = np.vstack([headshape, p["r"]*1000])  # Stack headshape points

    # Get the location to save the files to
    coreg_filenames = get_rhino_filenames(subjects_dir, subject)
    path_to_save    = coreg_filenames['coreg']['basedir']

    # Save to .txt files
    if nasion.any():
        np.savetxt(os.path.join(path_to_save, "polhemus_nasion.txt"), nasion.T, delimiter=' ', fmt='%.40e')
    if LPA.any():
        np.savetxt(os.path.join(path_to_save, "polhemus_LPA.txt"), LPA.T, delimiter=' ', fmt='%.40e')
    if RPA.any():
        np.savetxt(os.path.join(path_to_save, "polhemus_RPA.txt"), RPA.T, delimiter=' ', fmt='%.40e')
    if headshape.any():
        np.savetxt(os.path.join(path_to_save, "polhemus_headshape.txt"), headshape.T, delimiter=' ', fmt='%.40e')

    log_or_print("Saved files to {}".format(path_to_save))

def downsample_headshape(
    headshape,
    downsample_amount=0.015,    # average over 1.5cm
    include_facial_info=True,
    remove_zlim=0.02,           # Remove 2cm above nasion
    angle=10,                   # At an angle of 10deg
    method="gridaverage",
    face_Z = [-0.08, 0.02],     # Z-axis (up-down) -8cm to 2cm
    face_Y = [0.06, 0.15],      # Y-axis (forward-back) 6 to 15cm
    face_X = [-0.07, 0.07],     # X-axis (left_right) -7 to 7cm
    downsample_facial_info=True,
    downsample_facial_info_amount=0.008 # average over 0.8cm
):
    """
    Downsample headshape information for more accurate coregistration.

    Parameters
    ----------
    - headshape : numpy.ndarray
        Nx3 array of headshape points in meters.
    - include_facial_info : bool
        Include facial points if True.
    - remove_zlim : float
        Remove points above nasion on the z-axis in meters.
    - method : str
        Downsampling method. Note: only method supported is 'gridaverage'.
    - facial_info_above_z (float): float
        Max z-value for facial points in meters.
    - facial_info_below_z : float
        Min z-value for facial points in meters.
    - facial_info_above_y : float
        Max y-value for facial points in meters.
    - facial_info_below_y : float
        Min y-value for facial points in meters.
    - facial_info_below_x : float
        Min x-value for facial points in meters.
    - downsample_facial_info : bool
        Whether to downsample facial points.
    - downsample_facial_info_amount : float
        Grid size for downsampling facial info in meters.

    Returns:
    - numpy.ndarray: Downsampled headshape points.
    """
    # Original headshape copy
    headshape_orig = np.copy(headshape)

    # Filter facial points if needed
    if include_facial_info:
        facial_mask = (
            (headshape[:, 2] > face_Z[0]) &
            (headshape[:, 2] < face_Z[1]) &
            (headshape[:, 1] > face_Y[0]) &
            (headshape[:, 1] < face_Y[1]) &
            (headshape[:, 0] > face_X[0]) &
            (headshape[:, 0] < face_X[1])
        )
        # Separate facial points and non-facial points
        facial_points = headshape[facial_mask]

        # Print shape of the headshape without facial points
        if downsample_facial_info:
            facial_points = grid_average_downsample(facial_points, downsample_facial_info_amount)


    # Remove points above z-limit
    if remove_zlim is not None:
        log_or_print('Removing Points Below Z_lim')
        # Rotate the point cloud to align the Z-axis with the specified angle
        rotated_headshape = rotate_pointcloud(headshape, angle, 'x')
        
        # Filter points based on Z-limit
        z_mask = rotated_headshape[:, 2] > remove_zlim
        filtered_rotated_points = rotated_headshape[z_mask]
        
        # Rotate the filtered points back to the original orientation
        headshape = rotate_pointcloud(filtered_rotated_points, -angle, 'x')

    # Downsample the overall headshape
    if method == 'gridaverage':
        headshape = grid_average_downsample(headshape,downsample_amount)
    else:
        raise ValueError("Unsupported downsampling method: {}".format(method))

    # Combine headshape and facial points (if included)
    if include_facial_info:
        headshape = np.vstack((headshape, facial_points))

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(headshape_orig[:, 0], headshape_orig[:, 1], headshape_orig[:, 2], c='red', alpha=0.1, s=1)
    ax.scatter(headshape[:, 0], headshape[:, 1], headshape[:, 2], c='blue', alpha=1, s=10)
    plt.title("Downsampled Headshape")
    plt.show()

    return headshape


def rotate_pointcloud(points, angle_degrees, axis='x'):
    """
    Rotates the point cloud around a specified axis.

    Parameters
    ----------
    points : np.array
        Headshape points
    angle_degrees: float
        Amount to rotate in degrees.
    axis: str
        Axis to rotate.
    """
    angle_radians = np.radians(angle_degrees)
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
    
    return np.dot(points, rotation_matrix.T)


def grid_average_downsample(point_cloud, voxel_size):
    """
    Downsample a point cloud using grid averaging.

    This function divides the space into a voxel grid, computes the average 
    position of points within each voxel, and returns a downsampled point cloud.

    Parameters
    ----------
    point_cloud : numpy.ndarray
        A numpy array of shape (N, 3) representing the point cloud, where N 
        is the number of points, and each point has (x, y, z) coordinates.

    voxel_size : float
        The size of the voxel grid. Points within a grid cell are averaged 
        to compute the downsampled point.

    Returns
    -------
    downsampled_cloud : numpy.ndarray
        A numpy array of shape (M, 3) representing the downsampled point cloud, 
        where M is the number of voxels containing points.

    Notes
    -----
    - This method assumes the input point cloud is dense and unstructured.
    - For very large point clouds, consider optimizing memory usage.

    Examples
    --------
    >>> import numpy as np
    >>> point_cloud = np.random.rand(1000, 3)  # Generate random point cloud
    >>> voxel_size = 0.05
    >>> downsampled_cloud = grid_average_downsample(point_cloud, voxel_size)
    >>> print(downsampled_cloud.shape)
    """
    # Quantize the coordinates into voxel indices
    voxel_indices = np.floor(point_cloud / voxel_size).astype(np.int32)
    
    # Use a dictionary to collect points in the same voxel
    voxel_dict = {}
    for idx, point in zip(voxel_indices, point_cloud):
        key = tuple(idx)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(point)
    
    # Compute the average for each voxel
    downsampled_cloud = np.array([np.mean(voxel_dict[key], axis=0) for key in voxel_dict])
    
    return downsampled_cloud


def replace_headshape(raw, ds_headshape):
    """
    Replace headshape points in an MNE Raw object with downsampled points and update fiducials.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw object containing the original digitization points.
    ds_headshape : ndarray, shape (N, 3)
        Array of downsampled headshape points (in head coordinates).

    Returns
    -------
    raw_copy : mne.io.Raw
        A copy of the input Raw object with the updated headshape points and montage.
    """
    # Get current digitization points
    dig = raw.info['dig']

    # Initialize fiducial positions
    fid_positions = {
        'nasion': None,
        'lpa': None,
        'rpa': None
    }

    # Extract fiducials from the dig points
    for f in dig:
        if f['coord_frame'] == 4:  # Ensure it's in the head coordinate frame
            if f['ident'] == 2 and fid_positions['nasion'] is None:  # Nasion
                fid_positions['nasion'] = f['r']
            elif f['ident'] == 1 and fid_positions['lpa'] is None:  # Left Preauricular (LPA)
                fid_positions['lpa'] = f['r']
            elif f['ident'] == 3 and fid_positions['rpa'] is None:  # Right Preauricular (RPA)
                fid_positions['rpa'] = f['r']

    # Verify the extracted fiducials
    if any(v is None for v in fid_positions.values()):
        raise ValueError("One or more fiducials (nasion, LPA, RPA) not found in the head coordinate frame.")

    # Create a DigMontage using the extracted fiducials and downsampled headshape points
    montage = make_dig_montage(
        hsp=ds_headshape,  # Downsampled headshape points
        nasion=fid_positions['nasion'],
        lpa=fid_positions['lpa'],
        rpa=fid_positions['rpa'],
        coord_frame='head'
    )

    # Create a copy of the raw object and set the new montage
    raw_copy = raw.copy()
    raw_copy.set_montage(montage)

    return raw_copy

