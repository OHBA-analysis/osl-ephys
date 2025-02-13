"""Functions to handle parcellation.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import os.path as op
from pathlib import Path

import fsl
import mne
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from scipy.spatial import KDTree
from scipy.signal import welch
from nilearn.plotting import plot_markers
from mpl_toolkits.axes_grid1 import make_axes_locatable

import osl_ephys.source_recon.rhino.utils as rhino_utils
from osl_ephys.utils.logger import log_or_print
from osl_ephys.source_recon import freesurfer_utils

def load_parcellation(parcellation_file, freesurfer=False, subject=None):
    """Load a parcellation file.

    Parameters
    ----------
    parcellation_file : str
        Path to parcellation file.
    freesurfer : bool, optional
        Are we loading a FreeSurfer parcellation?
    subject : str
        Subject ID. Only needed for FreeSurfer parcellations.

    Returns
    -------
    parcellation : nibabel image or mne.Label
        Parcellation.
    """
    
    # is it a freesurfer parcellation
    if freesurfer:
        if subject is None:
            subject = "fsaverage"
        
        avail = mne.label._read_annot_cands(os.path.join(os.environ["SUBJECTS_DIR"], subject, 'label'))
        if parcellation_file in avail:
            labels = mne.label.read_labels_from_annot(subject, parcellation_file)
            if parcellation_file == 'aparc' or parcellation_file == "oasis.chubs":
                labels = [l for l in labels if "unknown" not in l.name]
            elif parcellation_file == 'aparc.a2009s':
                labels = [l for l in labels if "Unknown" not in l.name]
            elif parcellation_file == "Yeo2011_7Networks_N1000" or parcellation_file == "Yeo2011_17Networks_N1000":
                labels = [l for l in labels if "FreeSurfer_Defined_Medial_Wall" not in l.name]
            elif parcellation_file == "PALS_B12_Brodmann":
                labels = [l for l in labels if "Brodmann" in l.name]
            elif parcellation_file == "PALS_B12_Lobes":
                labels = [l for l in labels if "LOBE" in l.name] 
            return labels
    
    # otherwise, load the nifti parcellation file
    parcellation_file = find_file(parcellation_file, freesurfer=freesurfer)

    return nib.load(parcellation_file)


def _find_package_file(filename):
    files_dir = str(Path(__file__).parent) + "/files/"
    if op.exists(files_dir + filename):
        return files_dir + filename


def _find_freesurfer_file(filename):
    avail = mne.label._read_annot_cands(
        os.path.join(os.environ["SUBJECTS_DIR"], 'fsaverage', 'label')
    )
    if filename in avail:
        filename, hemis = mne.label._get_annot_fname(
            None, 'fsaverage', 'both', filename, os.environ['SUBJECTS_DIR']
        )
        return filename


def find_file(filename, freesurfer=False):
    """Look for a parcellation file within the package.

    Parameters
    ----------
    filename : str
        Path to parcellation file to look for.
    freesurfer : bool, optional
        Should we look in the freesurfer directory?

    Returns
    -------
    filename : str
        Path to parcellation file found.
    """
    if not op.exists(filename):
        if freesurfer:
            filename = _find_freesurfer_file(filename)
        else:
            filename = _find_package_file(filename)
        
    if filename is None:
        raise FileNotFoundError(filename)

    return filename


def guess_parcellation(data, return_path=False):
    """Guess parcellation file from data.
    
    Parameters
    ----------
    data : vector or matrix
        Data to guess parcellation from. first dimension is assumed to be parcels.
    return_path : bool
        If True, return path to parcellation file, otherwise return filename.
        
    returns
    -------
    filename : str
        Path to parcellation file.
    """
    if type(data) is int:
        nparc = data
    else:
        nparc = data.shape[0]
        
    # print('Guessing parcellation from data with {} parcels'.format(nparc))
    if nparc==52:
        fname = "Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"
    elif nparc==50:
        fname = "Glasser50_space-MNI152NLin6_res-8x8x8.nii.gz"
    elif nparc==38:
        fname = "fMRI_parcellation_ds8mm.nii.gz"
    elif nparc==39:
        fname = "fmri_d100_parcellation_with_PCC_tighterMay15_v2_8mm.nii.gz"
    elif nparc==78:
        fname = "aal_cortical_merged_8mm_stacked.nii.gz"
    
    # the following are FreeSurfer parcellations:
    elif nparc==68: # Desikan-Killiany (2006)
        fname = "aparc"
    # TODO: find out how many valid parcels these contain (might contain unkowns and ??? etc).
    # elif nparc==156:
    #     fname = "aparc.a2005s"
    elif nparc==144: # Destrieux et al. 2010
        fname = "aparc.a2009s"
    elif nparc==16:
        fname = "oasis.chubs"
    elif nparc==82:
        fname = "PALS_B12_Brodmann"
    elif nparc==10:
        fname = "PALS_B12_Lobes"
    # elif nparc==105:
    #     fname = "PALS_B12_OrbitoFrontal"
    # elif nparc==43:
    #     fname = "PALS_B12_Visuotopic"
    elif nparc==34:
        fname = "Yeo2011_17Networks_N1000"
    elif nparc==14:
        fname = "Yeo2011_7Networks_N1000"
    else:
        raise ValueError("Can't guess parcellation for {} channels".format(nparc))
    # print('Guessing parcellation is {}'.format(fname))
    if return_path:
        return find_file(fname)
    else:
        return find_file(fname).split('/')[-1]
 

def vol_parcellate_timeseries(parcellation_file, voxel_timeseries, voxel_coords, method, working_dir):
    """Parcellate a voxel time series.

    Parameters
    ----------
    parcellation_file : str
        Parcellation file (or path to parcellation file).
    voxel_timeseries : numpy.ndarray
        (nvoxels x ntpts) or (nvoxels x ntpts x ntrials) data to be parcellated.
        Data is assumed to be in same space as the parcellation (e.g. typically corresponds to the output from beamforming.transform_recon_timeseries).
    voxel_coords : numpy.ndarray
        (nvoxels x 3) coordinates of voxel_timeseries in mm in same space as parcellation (e.g. typically corresponds to the output from beamforming.transform_recon_timeseries).
    method : str
        ``'pca'`` - take 1st PC in each parcel
        ``'spatial_basis'`` - The parcel time-course for each spatial map is the 1st PC from all voxels, weighted by the spatial map.
        If the parcellation is unweighted and non-overlapping, 'spatialBasis' will give the same result as 'PCA' except with a different normalization.
    working_dir : str
        Directory to put temporary file in. If None, attempt to use same directory as passed in parcellation.

    Returns
    -------
    parcel_timeseries : numpy.ndarray
        nparcels x ntpts, or nparcels x ntpts x ntrials, parcellated data.
    voxel_weightings: numpy.ndarray
        nvoxels x nparcels, Voxel weightings for each parcel, corresponds to parcel_data = voxel_weightings.T * voxel_data
    voxel_assignments: bool numpy.ndarray
        nvoxels x nparcels, Boolean assignments indicating for each voxel the winner takes all parcel it belongs to.
    """
    parcellation_asmatrix = resample_parcellation(parcellation_file, voxel_coords, working_dir)
    return _get_parcel_timeseries(voxel_timeseries, parcellation_asmatrix, method=method)


def _get_parcel_timeseries(voxel_timeseries, parcellation_asmatrix, method="spatial_basis"):
    """Calculate parcel timeseries.

    Parameters
    ----------
    voxel_timeseries : numpy.ndarray
        (nvoxels x ntpts) or (nvoxels x ntpts x ntrials) and is assumed to be on the same grid as parcellation (typically output by beamforming.transform_recon_timeseries).
    parcellation_asmatrix: numpy.ndarray
        (nvoxels x nparcels) and is assumed to be on the same grid as voxel_timeseries.
    method : str
        ``'pca'`` - take 1st PC of voxels
        ``'spatial_basis'`` - The parcel time-course for each spatial map is the 1st PC from all voxels, weighted by the spatial map.
        If the parcellation is unweighted and non-overlapping, 'spatialBasis' will give the same result as 'PCA' except with a different normalization.

    Returns
    -------
    parcel_timeseries : numpy.ndarray
        nparcels x ntpts, or nparcels x ntpts x ntrials
    voxel_weightings : numpy.ndarray
        nvoxels x nparcels
        Voxel weightings for each parcel to compute parcel_timeseries from
        voxel_timeseries
    voxel_assignments : bool numpy.ndarray
        nvoxels x nparcels
        Boolean assignments indicating for each voxel the winner takes all
        parcel it belongs to
    """

    if parcellation_asmatrix.shape[0] != voxel_timeseries.shape[0]:
        Exception("Parcellation has {} voxels, but data has {}".format(parcellation_asmatrix.shape[0], voxel_timeseries.shape[0]))

    if len(voxel_timeseries.shape) == 2:
        # Add dim for trials
        voxel_timeseries = np.expand_dims(voxel_timeseries, axis=2)
        added_dim = True
    else:
        added_dim = False

    nparcels = parcellation_asmatrix.shape[1]
    ntpts = voxel_timeseries.shape[1]
    ntrials = voxel_timeseries.shape[2]

    # Combine the trials and time dimensions together, we will re-separate them after the parcel timeseries are computed
    voxel_timeseries_reshaped = np.reshape(voxel_timeseries, (voxel_timeseries.shape[0], ntpts * ntrials))
    parcel_timeseries_reshaped = np.zeros((nparcels, ntpts * ntrials))

    voxel_weightings = np.zeros(parcellation_asmatrix.shape)

    if method == "spatial_basis":
        # estimate temporal-STD of data for normalisation
        temporal_std = np.maximum(np.std(voxel_timeseries_reshaped, axis=1), np.finfo(float).eps)

        for pp in range(nparcels):
            # Scale group maps so all have a positive peak of height 1 in case there is a very noisy outlier, choose the sign from the top 5% of magnitudes
            thresh = np.percentile(np.abs(parcellation_asmatrix[:, pp]), 95)
            mapsign = np.sign(np.mean(parcellation_asmatrix[parcellation_asmatrix[:, pp] > thresh, pp]))
            scaled_parcellation = mapsign * parcellation_asmatrix[:, pp] / np.max(np.abs(parcellation_asmatrix[:, pp]))

            # Weight all voxels by the spatial map in question. Apply the mask first then weight to reduce memory use
            weighted_ts = voxel_timeseries_reshaped[scaled_parcellation > 0, :]
            weighted_ts = np.multiply(weighted_ts, np.reshape(scaled_parcellation[scaled_parcellation > 0], [-1, 1]))
            weighted_ts = weighted_ts - np.reshape(np.mean(weighted_ts, axis=1), [-1, 1])
            
            # Perform SVD and take scores of 1st PC as the node time-series
            #
            # U is nVoxels by nComponents - the basis transformation
            # S*V holds nComponents by time sets of PCA scores - the timeseries data in the new basis
            d, U = scipy.sparse.linalg.eigs(weighted_ts @ weighted_ts.T, k=1)
            U = np.real(U)
            d = np.real(d)
            S = np.sqrt(np.abs(np.real(d)))
            V = weighted_ts.T @ U / S
            pca_scores = S @ V.T

            # 0.5 is a decent arbitrary threshold used in fslnets after playing with various maps
            this_mask = scaled_parcellation[scaled_parcellation > 0] > 0.5

            if np.any(this_mask):  # the mask is non-zero
                # U is the basis by which voxels in the mask are weighted to form the scores of the 1st PC
                relative_weighting = np.abs(U[this_mask]) / np.sum(np.abs(U[this_mask]))
                ts_sign = np.sign(np.mean(U[this_mask]))
                ts_scale = np.dot(np.reshape(relative_weighting, [-1]), temporal_std[scaled_parcellation > 0][this_mask])

                node_ts = ts_sign * (ts_scale / np.maximum(np.std(pca_scores), np.finfo(float).eps)) * pca_scores

                inds = np.where(scaled_parcellation > 0)[0]
                voxel_weightings[inds, pp] = (
                    ts_sign * ts_scale / np.maximum(np.std(pca_scores), np.finfo(float).eps) * (np.reshape(U, [-1]) * scaled_parcellation[scaled_parcellation > 0].T)
                )

            else:
                log_or_print(
                    "WARNING: An empty parcel mask was found for parcel {} ".format(pp)
                    + "when calculating its time-courses\n"
                    + "The parcel will have a flat zero time-course.\n"
                    + "Check this does not cause further problems with the analysis.\n"
                )

                node_ts = np.zeros(ntpts * ntrials)
                inds = np.where(scaled_parcellation > 0)[0]
                voxel_weightings[inds, pp] = 0

            parcel_timeseries_reshaped[pp, :] = node_ts

    elif method == "pca":
        log_or_print(
            "PCA assumes a binary parcellation.\n"
            "Parcellation will be binarised if it is not already (any voxels >0 are set to 1, otherwise voxels are set to 0), i.e. any weightings will be ignored.\n"
        )

        # Check that each voxel is only a member of one parcel
        if any(np.sum(parcellation_asmatrix, axis=1) > 1):
            log_or_print("WARNING: Each voxel is meant to be a member of at most one parcel, when using the PCA method.\nResults may not be sensible")

        # Estimate temporal-STD of data for normalisation
        temporal_std = np.maximum(np.std(voxel_timeseries_reshaped, axis=1), np.finfo(float).eps)

        # Perform PCA on each parcel and select 1st PC scores to represent parcel
        for pp in range(nparcels):
            if any(parcellation_asmatrix[:, pp]):  # non-zero
                parcel_data = voxel_timeseries_reshaped[parcellation_asmatrix[:, pp] > 0, : ]
                parcel_data = parcel_data - np.reshape(np.mean(parcel_data, axis=1), [-1, 1])

                # Perform svd and take scores of 1st PC as the node time-series
                #
                # U is nVoxels by nComponents - the basis transformation
                # S*V holds nComponents by time sets of PCA scores - the timeseries data in the new basis
                d, U = scipy.sparse.linalg.eigs(parcel_data @ parcel_data.T, k=1)
                U = np.real(U)
                d = np.real(d)
                S = np.sqrt(np.abs(np.real(d)))
                V = parcel_data.T @ U / S
                pca_scores = S @ V.T

                # Restore sign and scaling of parcel time-series
                # U indicates the weight with which each voxel in the parcel contributes to the 1st PC
                relative_weighting = np.abs(U) / np.sum(np.abs(U))
                ts_sign = np.sign(np.mean(U))
                ts_scale = np.dot(np.reshape(relative_weighting, [-1]), temporal_std[parcellation_asmatrix[:, pp] > 0])

                node_ts = (ts_sign * ts_scale / np.maximum(np.std(pca_scores), np.finfo(float).eps)) * pca_scores

                inds = np.where(parcellation_asmatrix[:, pp] > 0)[0]
                voxel_weightings[inds, pp] = ts_sign * ts_scale / np.maximum(np.std(pca_scores), np.finfo(float).eps) * np.reshape(U, [-1])

            else:
                log_or_print(
                    "WARNING: An empty parcel mask was found for parcel {}".format(pp)
                    + " when calculating its time-courses\n"
                    + "The parcel will have a flat zero time-course.\n"
                    + "Check this does not cause further problems with the analysis.\n"
                )

                node_ts = np.zeros(ntpts * ntrials)
                inds = np.where(parcellation_asmatrix[:, pp] > 0)[0]
                voxel_weightings[inds, pp] = 0

            parcel_timeseries_reshaped[pp, :] = node_ts

    else:
        Exception("Invalid method specified")

    # Re-separate the trials and time dimensions
    parcel_timeseries = np.reshape(parcel_timeseries_reshaped, (nparcels, ntpts, ntrials))
    if added_dim:
        parcel_timeseries = np.squeeze(parcel_timeseries, axis=2)

    # Compute voxel_assignments using winner takes all
    voxel_assignments = np.zeros(voxel_weightings.shape)
    for ivoxel in range(voxel_weightings.shape[0]):
        winning_parcel = np.argmax(voxel_weightings[ivoxel, :])
        voxel_assignments[ivoxel, winning_parcel] = 1

    return parcel_timeseries, voxel_weightings, voxel_assignments


def surf_parcellate_timeseries(subject_dir, subject, stc, method, parc):
    """Save parcellated data as a fif file.
    
    Parameters
    ----------
    subject_dir : str
        Path to subject directory.
    subject : str
        Subject ID.
    stc : mne.SourceEstimate
        Source estimate.
    method : str
        Parcellation method. Can be 'pca_flip', 'max', 'mean', 'mean_flip', 'auto'
    parc : str
        Parcellation name.
    """
    fs_files = freesurfer_utils.get_freesurfer_filenames(subject_dir, subject)

    labels = load_parcellation(parc, freesurfer=True, subject=subject)

    src = mne.read_source_spaces(fs_files['coreg']['source_space'])
    parcel_data = mne.extract_label_time_course(stc, labels, src, mode=method)

    return parcel_data


def resample_parcellation(parcellation_file, voxel_coords, working_dir=None):
    """Resample parcellation so that its voxel coords correspond (using nearest neighbour) to passed in voxel_coords.
    Passed in voxel_coords and parcellation must be in the same space, e.g. MNI.

    Used to make sure that the parcellation's voxel coords are the same as the voxel coords for some timeseries data, before calling _get_parcel_timeseries.

    Parameters
    ----------
    parcellation_file : str
        Path to parcellation file. In same space as voxel_coords.
    voxel_coords :
        (nvoxels x 3) coordinates in mm in same space as parcellation.
    working_dir : str
        Dir to put temp file in. If None, attempt to use same dir as passed in parcellation.

    Returns
    -------
    parcellation_asmatrix : numpy.ndarray
        (nvoxels x nparcels) resampled parcellation
    """
    gridstep = int(rhino_utils.get_gridstep(voxel_coords.T) / 1000)
    log_or_print(f"gridstep = {gridstep} mm")

    parcellation_file = find_file(parcellation_file)
    path, parcellation_name = op.split(op.splitext(op.splitext(parcellation_file)[0])[0])

    if working_dir is None:
        working_dir = path
    else:
        os.makedirs(working_dir, exist_ok=True)

    parcellation_resampled = op.join(working_dir, parcellation_name + "_{}mm.nii.gz".format(gridstep))

    # Create standard brain of the required resolution
    #
    # Command: flirt -in <parcellation_file> -ref <parcellation_file> -out <parcellation_resampled> -applyisoxfm <gridstep>
    #
    # Note, this call raises:
    #
    #   Warning: An input intended to be a single 3D volume has multiple timepoints. Input will be truncated to first volume,
    #   but this functionality is deprecated and will be removed in a future release.
    #
    # However, it doesn't look like the input be being truncated, the resampled parcellation appears to be a 4D volume.
    fsl.wrappers.flirt(parcellation_file, parcellation_file, out=parcellation_resampled, applyisoxfm=gridstep)

    nparcels = nib.load(parcellation_resampled).get_fdata().shape[3]

    # parcellation_asmatrix will be the parcels mapped onto the same dipole grid as voxel_coords
    parcellation_asmatrix = np.zeros((voxel_coords.shape[1], nparcels))

    for parcel_index in range(nparcels):
        parcellation_coords, parcellation_vals = rhino_utils.niimask2mmpointcloud(parcellation_resampled, parcel_index)

        kdtree = KDTree(parcellation_coords.T)

        # Find each voxel_coords best matching parcellation_coords and assign the corresponding parcel value to
        for ind in range(voxel_coords.shape[1]):
            distance, index = kdtree.query(voxel_coords[:, ind])

            # Exclude from parcel any voxel_coords that are further than gridstep away from the best matching parcellation_coords
            if distance < gridstep:
                parcellation_asmatrix[ind, parcel_index] = parcellation_vals[index]

    return parcellation_asmatrix


def symmetric_orthogonalise(timeseries, maintain_magnitudes=False, compute_weights=False):
    """Returns orthonormal matrix L which is closest to A, as measured by the Frobenius norm of (L-A). The orthogonal matrix is constructed from a singular
    value decomposition of A.

    If maintain_magnitudes is True, returns the orthogonal matrix L, whose columns have the same magnitude as the respective columns of A, and which is closest to
    A, as measured by the Frobenius norm of (L-A).

    Parameters
    ----------
    timeseries : numpy.ndarray
        (nparcels x ntpts) or (nparcels x ntpts x ntrials) data to orthoganlise. In the latter case, the ntpts and ntrials dimensions are concatenated.
    maintain_magnitudes : bool
    compute_weights : bool

    Returns
    -------
    ortho_timeseries : numpy.ndarray
        (nparcels x ntpts) or (nparcels x ntpts x ntrials) orthoganalised data
    weights : numpy.ndarray
        (optional output depending on compute_weights flag) weighting matrix such that, ortho_timeseries = timeseries * weights

    References
    ----------
    Colclough, G. L., Brookes, M., Smith, S. M. and Woolrich, M. W., "A symmetric multivariate leakage correction for MEG connectomes," NeuroImage 117, pp. 439-448 (2015)
    """

    if len(timeseries.shape) == 2:
        # add dim for trials:
        timeseries = np.expand_dims(timeseries, axis=2)
        added_dim = True
    else:
        added_dim = False

    nparcels = timeseries.shape[0]
    ntpts = timeseries.shape[1]
    ntrials = timeseries.shape[2]
    compute_weights = False

    # combine the trials and time dimensions together,
    # we will re-separate them after the parcel timeseries are computed
    timeseries = np.transpose(np.reshape(timeseries, (nparcels, ntpts * ntrials)))

    if maintain_magnitudes:
        D = np.diag(np.sqrt(np.diag(np.transpose(timeseries) @ timeseries)))
        timeseries = timeseries @ D

    [U, S, V] = np.linalg.svd(timeseries, full_matrices=False)

    # we need to check that we have sufficient rank
    tol = max(timeseries.shape) * S[0] * np.finfo(type(timeseries[0, 0])).eps
    r = sum(S > tol)
    full_rank = r >= timeseries.shape[1]

    if full_rank:
        # polar factors of A
        ortho_timeseries = U @ np.conjugate(V)
    else:
        raise ValueError("Not full rank, rank required is {}, but rank is only {}".format(timeseries.shape[1], r))

    if compute_weights:
        # weights are a weighting matrix such that,
        # ortho_timeseries = timeseries * weights
        weights = np.transpose(V) @ np.diag(1.0 / S) @ np.conjugate(V)

    if maintain_magnitudes:
        # scale result
        ortho_timeseries = ortho_timeseries @ D

        if compute_weights:
            # weights are a weighting matrix such that,
            # ortho_timeseries = timeseries * weights
            weights = D @ weights @ D

    # Re-separate the trials and time dimensions
    ortho_timeseries = np.reshape(np.transpose(ortho_timeseries), (nparcels, ntpts, ntrials))

    if added_dim:
        ortho_timeseries = np.squeeze(ortho_timeseries, axis=2)

    if compute_weights:
        return ortho_timeseries, weights
    else:
        return ortho_timeseries


def parcel_centers(parcellation_file):
    """Get coordinates of parcel centers.

    Parameters
    ----------
    parcellation_file : str
        Path to parcellation file.

    Returns
    -------
    coords : np.ndarray
        Coordinates of each parcel. Shape is (n_parcels, 3).
    """
    parcellation = load_parcellation(parcellation_file)
    if isinstance(parcellation, nib.nifti1.Nifti1Image):
        n_parcels = parcellation.shape[3]
        data = parcellation.get_fdata()
        nonzero = [np.nonzero(data[..., i]) for i in range(n_parcels)]
        nonzero_coords = [nib.affines.apply_affine(parcellation.affine, np.array(nz).T) for nz in nonzero ]
        weights = [data[..., i][nz] for i, nz in enumerate(nonzero)]
        coords = np.array([np.average(c, weights=w, axis=0) for c, w in zip(nonzero_coords, weights)])
    elif isinstance(parcellation, list) and isinstance(parcellation[0], mne.Label): # freesurfer parcellation
        vertices = np.array([l.center_of_mass() for l in parcellation])
        hemis = [0 if l.hemi=='lh' else 1 for l in parcellation]
        coords = mne.vertex_to_mni(vertices, hemis, "fsaverage")
    return coords


def plot_parcellation(parcellation_file, **kwargs):
    """Plots a parcellation.

    Parameters
    ----------
    parcellation_file : str
        Path to parcellation file.
    kwargs : keyword arguments
        Keyword arguments to pass to nilearn.plotting.plot_markers.
    """
    parc_centers = parcel_centers(parcellation_file)
    n_parcels = parc_centers.shape[0]
    return plot_markers(np.zeros(n_parcels), parc_centers, colorbar=False, node_cmap="binary_r", **kwargs)


def plot_psd(parc_ts, fs, parcellation_file, filename, freq_range=None):
    """Plot PSD of each parcel time course.

    Parameters
    ---------- 
    parc_ts : np.ndarray
        (parcels, time) or (parcels, time, epochs) time series.
    fs : float
        Sampling frequency in Hz.
    parcellation_file : str
        Path to parcellation file.
    filename : str
        Output filename.
    freq_range : list of len 2
        Low and high frequency in Hz.
    """
    if parc_ts.ndim == 3:
        # Calculate PSD for each epoch individually and average
        psd = []
        for i in range(parc_ts.shape[-1]):
            f, p = welch(parc_ts[..., i], fs=fs)
            psd.append(p)
        psd = np.mean(psd, axis=0)
    else:
        # Calcualte PSD of continuous data
        f, psd = welch(parc_ts, fs=fs)

    n_parcels = psd.shape[0]

    if freq_range is None:
        freq_range = [f[0], f[-1]]

    # Re-order to use colour to indicate anterior->posterior location
    parc_centers = parcel_centers(parcellation_file)
    order = np.argsort(parc_centers[:, 1])
    parc_centers = parc_centers[order]
    psd = psd[order]

    # Plot PSD
    fig, ax = plt.subplots()
    cmap = plt.cm.viridis_r
    for i in reversed(range(n_parcels)):
        ax.plot(f, psd[i], c=cmap(i / n_parcels))
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("PSD (a.u.)", fontsize=14)
    ax.set_xlim(freq_range[0], freq_range[1])
    ax.tick_params(axis="both", labelsize=14)
    plt.tight_layout()

    # Plot parcel topomap
    inside_ax = ax.inset_axes([0.45, 0.55, 0.5, 0.55])
    plot_markers(np.arange(n_parcels), parc_centers, node_size=12, colorbar=False, axes=inside_ax)

    # Save
    log_or_print(f"saving {filename}")
    plt.savefig(filename)
    plt.close()


def plot_correlation(parc_ts, filename):
    """Plot correlation between parcel time courses.

    Parameters
    ----------
    parc_ts : np.ndarray
        (parcels, time) or (parcels, time, epochs) time series.
    filename : str
        Output filename.
    """
    if parc_ts.ndim == 3:
        # (parcels, time, epochs) -> (parcels, time)
        shape = parc_ts.shape
        parc_ts = parc_ts.reshape(shape[0], shape[1] * shape[2])

    # Calculate correlation
    corr = np.corrcoef(parc_ts)
    np.fill_diagonal(corr, 0)

    # Plot
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(corr)
    ax.set_xlabel("Parcel", fontsize=14)
    ax.set_ylabel("Parcel", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Save
    log_or_print(f"saving {filename}")
    plt.savefig(filename)
    plt.close(fig)


def _parcel_timeseries2nii(
    parcellation_file,
    parcel_timeseries_data,
    voxel_weightings,
    voxel_assignments,
    voxel_coords,
    out_nii_fname=None,
    working_dir=None,
    times=None,
    method="assignments",
):
    """Outputs parcel_timeseries_data as a niftii file using passed in parcellation.

    The parcellation and parcel_timeseries_data need to have the same number of parcels.

    Parameters
    ----------
    parcellation_file : str
        Path to parcellation file.
    parcel_timeseries_data: numpy.ndarray
        Needs to be nparcels x ntpts
    voxel_weightings : numpy.ndarray
        (nvoxels x nparcels) voxel weightings for each parcel to compute parcel_timeseries from voxel_timeseries.
    voxel_assignments : bool numpy.ndarray
        (nvoxels x nparcels) boolean assignments indicating for each voxel the winner takes all parcel it belongs to.
    voxel_coords : numpy.ndarray
        (nvoxels x 3) coordinates of voxel_timeseries in mm in same space as parcellation (e.g. typically corresponds to the output from beamforming.transform_recon_timeseries).
    working_dir : str
        Directory name to put files in.
    out_nii_fname : str
        Output name to put files in.
    times : array
        (ntpts,) times points in seconds. Will assume that time points are regularly spaced. Used to set nii file up correctly.
    method : str
        "weights" or "assignments"

    Returns
    -------
    out_nii_fname : str
        Output nii filename, will be output at spatial resolution of parcel_timeseries['voxel_coords'].
    """
    parcellation_file = find_file(parcellation_file)
    path, parcellation_name = op.split(op.splitext(op.splitext(parcellation_file)[0])[0])

    if working_dir is None:
        working_dir = path

    if out_nii_fname is None:
        out_nii_fname = op.join(working_dir, parcellation_name + "_timeseries.nii.gz")

    # compute parcellation_mask_file to be mean over all parcels
    parcellation_mask_file = op.join(working_dir, parcellation_name + "_mask.nii.gz")
    rhino_utils.system_call("fslmaths {} -Tmean {}".format(parcellation_file, parcellation_mask_file))

    if len(parcel_timeseries_data.shape) == 1:
        parcel_timeseries_data = np.reshape(parcel_timeseries_data, [parcel_timeseries_data.shape[0], 1])

    # Compute nmaskvoxels x ntpts voxel_data
    if method == "assignments":
        weightings = voxel_assignments
    elif method == "weights":
        weightings = np.linalg.pinv(voxel_weightings.T)
    else:
        raise ValueError("Invalid method. Must be assignments or weights.")

    voxel_data = weightings @ parcel_timeseries_data

    # voxel_coords is nmaskvoxels x 3 in mm
    gridstep = int(rhino_utils.get_gridstep(voxel_coords.T) / 1000)

    # Sample parcellation_mask to the desired resolution
    path, ref_brain_name = op.split(op.splitext(op.splitext(parcellation_mask_file)[0])[0])
    parcellation_mask_resampled = op.join(working_dir, ref_brain_name + "_{}mm_brain.nii.gz".format(gridstep))

    # create std brain of the required resolution
    rhino_utils.system_call("flirt -in {} -ref {} -out {} -applyisoxfm {}".format(parcellation_mask_file, parcellation_mask_file, parcellation_mask_resampled, gridstep))

    parcellation_mask_coords, vals = rhino_utils.niimask2mmpointcloud(parcellation_mask_resampled)
    parcellation_mask_inds = rhino_utils.niimask2indexpointcloud(parcellation_mask_resampled)

    vol = nib.load(parcellation_mask_resampled).get_fdata()
    vol = np.zeros(np.append(vol.shape[:3], parcel_timeseries_data.shape[1]))
    kdtree = KDTree(parcellation_mask_coords.T)

    # Find each voxel_coords best matching parcellation_mask_coords
    for ind in range(voxel_coords.shape[1]):
        distance, index = kdtree.query(voxel_coords[:, ind])
        # Exclude from parcel any voxel_coords that are further than gridstep away
        if distance < gridstep:
            vol[parcellation_mask_inds[0, index], parcellation_mask_inds[1, index], parcellation_mask_inds[2, index], :] = voxel_data[ind, :]

    # Save as nifti
    vol_nii = nib.Nifti1Image(vol, nib.load(parcellation_mask_resampled).affine)

    vol_nii.header.set_xyzt_units(2)  # mm
    if times is not None:
        vol_nii.header["pixdim"][4] = times[1] - times[0]
        vol_nii.header["toffset"] = 0
        vol_nii.header.set_xyzt_units(2, 8)  # mm and secs

    nib.save(vol_nii, out_nii_fname)

    return out_nii_fname


def convert2niftii(parc_data, parcellation_file, mask_file, tres=1, tmin=0):
    """Convert parcellation to NIfTI.

    Takes (nparcels) or (nvolumes x nparcels) parc_data and returns (xvoxels x yvoxels x zvoxels x nvolumes) niftii file containing parc_data on a volumetric grid.

    Parameters
    ----------
    parc_data : np.ndarray
        (nparcels) or (nvolumes x nparcels) parcel data.
    parcellation_file : str
        Path to niftii parcellation file.
    mask_file : str
        Path to niftii parcellation mask file.
    tres : float
        Resolution of 4th dimension in secs
    tmin : float
        Value of first time point in secs

    Returns
    -------
    nii : nib.Nifti1Image
        (xvoxels x yvoxels x zvoxels x nvolumes) nib.Nifti1Image containing parc_data on a volumetric grid.
    """

    if len(parc_data.shape) == 1:
        parc_data = np.reshape(parc_data, [1, -1])

    # Find files within the package
    parcellation_file = find_file(parcellation_file)
    mask_file = find_file(mask_file)

    # Load the mask
    mask = nib.load(mask_file)
    mask_grid = mask.get_fdata()
    mask_grid = mask_grid.ravel(order="F")

    # Get indices of non-zero elements, i.e. those which contain the brain
    non_zero_voxels = mask_grid != 0

    # Load the parcellation
    parcellation = nib.load(parcellation_file)
    parcellation_grid = parcellation.get_fdata()

    # Make a 2D array of voxel weights for each parcel
    n_parcels = parcellation.shape[-1]

    # check parcellation is compatible:
    if parc_data.shape[1] is not n_parcels:
        Exception("parcellation_file has a different number of parcels to the maps")

    voxel_weights = parcellation_grid.reshape(-1, n_parcels, order="F")

    # check mask is compatible with parcellation:
    if voxel_weights.shape[0] != mask_grid.shape[0]:
        Exception("parcellation_file has a different number of voxels to mask_file")

    voxel_weights = voxel_weights[non_zero_voxels]

    # Normalise the voxels weights
    voxel_weights /= voxel_weights.max(axis=0)[np.newaxis, ...]

    # Generate a spatial map vector for each mode
    n_voxels = voxel_weights.shape[0]
    n_modes = parc_data.shape[0]
    spatial_map_values = np.empty([n_voxels, n_modes])

    for i in range(n_modes):
        spatial_map_values[:, i] = voxel_weights @ parc_data[i]

    # Final spatial map as a 3D grid for each mode
    spatial_map = np.zeros([mask_grid.shape[0], n_modes])
    spatial_map[non_zero_voxels] = spatial_map_values
    spatial_map = spatial_map.reshape(mask.shape[0], mask.shape[1], mask.shape[2], n_modes, order="F")
    nii = nib.Nifti1Image(spatial_map, mask.affine, mask.header)

    nii.header["pixdim"][4] = tres
    nii.header["toffset"] = tmin

    return nii


def convert2mne_raw(parc_data, raw, parcel_names=None, extra_chans="stim"):
    """Create and returns an MNE raw object that contains parcellated data.

    Parameters
    ----------
    parc_data : np.ndarray
        (nparcels x ntpts) parcel data.
    raw : mne.Raw
        mne.io.raw object that produced parc_data via source recon and parcellation. Info such as timings and bad segments will be copied from this to parc_raw.
    parcel_names : list of str
        List of strings indicating names of parcels. If None then names are set to be parcel_0,...,parcel_{n_parcels-1}.
    extra_chans : str or list of str
        Extra channels, e.g. 'stim' or 'emg', to include in the parc_raw object. Defaults to 'stim'. stim channels are always added to parc_raw if they are present in raw.

    Returns
    -------
    parc_raw : mne.Raw
        Generated parcellation in mne.Raw format.
    """
    # What extra channels should we add to the parc_raw object?
    if isinstance(extra_chans, str):
        extra_chans = [extra_chans]
    extra_chans = np.unique(["stim"] + extra_chans)

    # parc_data is missing bad segments
    # We insert these before creating the new MNE object
    _, times = raw.get_data(reject_by_annotation="omit", return_times=True)
    indices = raw.time_as_index(times, use_rounding=True)
    data = np.zeros([parc_data.shape[0], len(raw.times)], dtype=np.float32)
    data[:, indices] = parc_data

    # Create Info object
    info = raw.info
    if parcel_names is None:
        parcel_names = [f"parcel_{i}" for i in range(data.shape[0])]
    parc_info = mne.create_info(ch_names=parcel_names, ch_types="misc", sfreq=info["sfreq"])

    # Create Raw object
    parc_raw = mne.io.RawArray(data, parc_info)

    # Copy timing info
    parc_raw.set_meas_date(raw.info["meas_date"])
    parc_raw.__dict__["_first_samps"] = raw.__dict__["_first_samps"]
    parc_raw.__dict__["_last_samps"] = raw.__dict__["_last_samps"]
    parc_raw.__dict__["_cropped_samp"] = raw.__dict__["_cropped_samp"]

    # Copy annotations from raw
    parc_raw.set_annotations(raw._annotations)

    # Add extra channels
    if "stim" not in raw:
        log_or_print("No stim channel to add to parc-raw.fif", warning=True)
    for extra_chan in extra_chans:
        if extra_chan in raw:
            chan_raw = raw.copy().pick(extra_chan)
            chan_data = chan_raw.get_data()
            chan_info = mne.create_info(chan_raw.ch_names, raw.info["sfreq"], [extra_chan] * chan_data.shape[0])
            chan_raw = mne.io.RawArray(chan_data, chan_info)
            parc_raw.add_channels([chan_raw], force_update_info=True)

    # Copy the description from the sensor-level Raw object
    parc_raw.info["description"] = raw.info["description"]

    return parc_raw


def convert2mne_epochs(parc_data, epochs, parcel_names=None):
    """Create and returns an MNE Epochs object that contains parcellated data.

    Parameters
    ----------
    parc_data : np.ndarray
        (nparcels x ntpts x epochs) parcel data.
    epochs : mne.Epochs
        mne.io.raw object that produced parc_data via source recon and parcellation. Info such as timings and bad segments will be copied from this to parc_raw.
    parcel_names : list of str
        List of strings indicating names of parcels. If None then names are set to be parcel_0,...,parcel_{n_parcels-1}.

    Returns
    -------
    parc_epo : mne.Epochs
        Generated parcellation in :py:class: mne.Epochs` format.
    """

    # Epochs info
    info = epochs.info

    # Create parc info
    if parcel_names is None:
        parcel_names = [f"parcel_{i}" for i in range(parc_data.shape[0])]

    parc_info = mne.create_info(ch_names=parcel_names, ch_types="misc", sfreq=info["sfreq"])
    parc_events = epochs.events

    # Parcellated data Epochs object
    parc_epo = mne.EpochsArray(np.swapaxes(parc_data.T, 1, 2), parc_info, parc_events)

    # Copy the description from the sensor-level Epochs object
    parc_epo.info["description"] = epochs.info["description"]

    return parc_epo


def spatial_dist_adjacency(parcellation_file, dist, verbose=False):
    """Compute adjacency from distances between parcels.

    Parameters
    ----------
    parcellation_file : str
        Path to parcellation file.
    dist : float
        Maximum (geodesic) distance in mm for two parcels to within to be considered as neighbours.
    verbose : bool
        Should we print the distance between parcels that are considered neighbours?

    Returns
    -------
    adj_mat : np.ndarray
        (n_parcels, n_parcels) matrix of zeros (indicating not neighbours) and ones (indicating parcels are neighbours).
    """

    # Function to calculate distance between 2 points based on their 3D coordinates
    distance = lambda x, y: np.sqrt(np.sum((x - y) ** 2))

    # Get coordinate of the centroid of each parcel
    coords = parcel_centers(parcellation_file)
    n_parcels = coords.shape[0]

    # Compute adjacency matrix
    adj_mat = np.zeros([n_parcels, n_parcels])
    for i in range(n_parcels):
        adj_mat[i, i] = 1
        for j in range(i + 1, n_parcels):
            d = distance(coords[i], coords[j])  # in mm
            if d < dist:
                if verbose:
                    print(f"parcels {i}, {j} : {np.round(d)} mm")
                adj_mat[i, j] = 1
                adj_mat[j, i] = 1

    return adj_mat


def parcel_vector_to_voxel_grid(mask_file, parcellation_file, vector):
    """Takes a vector of parcel values and return a 3D voxel grid.

    Parameters
    ----------
    mask_file : str
        Mask file for the voxel grid. Must be a NIFTI file.
    parcellation_file : str
        Parcellation file. Must be a NIFTI file.
    vector : np.ndarray
        Value at each parcel. Shape must be (n_parcels,).

    Returns
    -------
    voxel_grid : np.ndarray
        Value at each voxel. Shape is (x, y, z), where :code:`x`,
        :code:`y` and :code:`z` correspond to 3D voxel locations.
    """
    # Validation
    mask_file = find_file(mask_file)
    parcellation_file = find_file(parcellation_file)

    # Load the mask
    mask = nib.load(mask_file)
    mask_grid = mask.get_fdata()
    mask_grid = mask_grid.ravel(order="F")

    # Get indices of non-zero elements, i.e. those which contain the brain
    non_zero_voxels = mask_grid != 0

    # Load the parcellation
    parcellation = nib.load(parcellation_file)
    parcellation_grid = parcellation.get_fdata()

    # Make a 2D array of voxel weights for each parcel
    n_parcels = parcellation.shape[-1]

    # Check parcellation is compatible
    if vector.shape[0] != n_parcels:
        raise ValueError("parcellation_file has a different number of parcels to the vector")

    voxel_weights = parcellation_grid.reshape(-1, n_parcels, order="F")[non_zero_voxels]

    # Normalise the voxels weights
    voxel_weights /= voxel_weights.max(axis=0, keepdims=True)

    # Generate a vector containing value at each voxel
    voxel_values = voxel_weights @ vector

    # Final 3D voxel grid
    voxel_grid = np.zeros(mask_grid.shape[0])
    voxel_grid[non_zero_voxels] = voxel_values
    voxel_grid = voxel_grid.reshape(mask.shape[0], mask.shape[1], mask.shape[2], order="F")

    return voxel_grid

def convert2source_estimate(subjects_dir, data, parc=None, reference_brain='fsaverage'):
    """ Convert parcellated data to a source estimate.
    
    Parameters
    ----------
    subjects_dir : str
        Path to subjects directory.
    data : mne.Evoked or mne.Epochs
        Data to convert.
    parc : str
        Parcellation name.
    reference_brain : str
        Reference brain. Default is 'fsaverage'.
        
    Returns
    -------
    stc : mne.SourceEstimate
        Source estimate.    
    """
    os.environ["SUBJECTS_DIR"] = subjects_dir
    
    if reference_brain is None:
            subject = "fsaverage"
              
    if parc is None:
        parc = guess_parcellation(np.ones(data.get_data(picks='misc').shape[0]))
    
    labels = load_parcellation(parc)
    nparc=len(labels)
    
    src = mne.read_source_spaces(freesurfer_utils.get_freesurfer_filenames(os.environ["SUBJECTS_DIR"], reference_brain)['source_space'])
    
    vertices = [s["vertno"] for s in src]
    kernel = np.zeros((src[0]['nuse'], nparc))
    for i, l in enumerate(labels):
        v = mne.source_space.label_src_vertno_sel(l, src)[0]
        v = v[np.argmax([len(iv) for iv in v])]
        kernel[v, i] = 1
    
    return mne.SourceEstimate((kernel, data.get_data(picks='misc')), vertices, tmin=0, tstep=1/data.info['sfreq'])   
