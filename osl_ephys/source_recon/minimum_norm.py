"""Minimum norm source localization using MNE-Python.

"""
    
# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>
#          Hongyu Qian <hongyu.qian@queens.ox.ac.uk>


import os
import os.path as op

import mne
import numpy as np
from mne import (
    read_forward_solution,
    Covariance,
    compute_covariance,
    compute_raw_covariance,
)

from . import freesurfer_utils
from ..utils.logger import log_or_print

from osl_ephys.source_recon.rhino import utils as rhino_utils

import logging

logger = logging.getLogger(__name__)


def get_mne_filenames(subjects_dir, subject):
    """Get minimum norm (MNE) filenames.

    Files will be in subjects_dir/subject/mne

    Parameters
    ----------
    subjects_dir : string
        Directory containing the subject directories.
    subject : string
        Subject name.

    Returns
    -------
    filenames : dict
        A dict of files.
    """
    basedir = op.join(subjects_dir, subject, "mne")
    if " " in basedir:
        raise ValueError("subjects_dir cannot contain spaces.")
    os.makedirs(basedir, exist_ok=True)

    filenames = {
        "inverse_operator": op.join(basedir, "op-inv.fif"),
        "source_estimate_raw": op.join(basedir, "src-raw"), # followed by -lh/rh.stc
        "source_estimate_epo": op.join(basedir, "src-epo"),
    }
    return filenames


def create_inverse_operator(
    fwd,
    data,
    chantypes,
    rank,
    depth,
    loose,
    filename,
):
    """Creates minimum norm (MNE) inverse operator.
    
    Parameters
    ----------
    fwd : mne forward model or str
        Forward model.
    data : mne.io.Raw, mne.Epochs  
        Preprocessed data.
    chantypes : list
        List of channel types to include.
    rank : int
        Rank of the data covariance matrix.
    depth : float
        Depth weighting.
    loose : float
        Loose parameter.
    inv_op_filename : str
        Output filename.
    """
    log_or_print("*** RUNNING MNE SOURCE LOCALIZATION ***")
    
    noise_cov = calc_noise_cov(data, rank, chantypes)
    
    if isinstance(fwd, str):
        fwd = mne.read_forward_solution(fwd)

    inverse_operator = mne.minimum_norm.make_inverse_operator(
        data.info,
        fwd,
        noise_cov,
        loose=loose,
        depth=depth,
        rank=rank,
    )

    mne.minimum_norm.write_inverse_operator(
        filename, inverse_operator, overwrite=True
    )

    return inverse_operator
    

def apply_inverse_operator_surf(
    outdir,
    subject,
    data,
    method,
    lambda2,
    pick_ori,
    inverse_operator=None,
    morph="fsaverage",
    save=False,
):
    """Apply previously computed minimum norm inverse solution (surface).
    
    Parameters
    ----------
    outdir : str
        Output directory.
    subject : str
        Subject ID.
    data : mne.io.Raw, mne.Epochs
        Raw or Epochs object.
    inverse_operator : mne.minimum_norm.InverseOperator
        Inverse operator.
    method : str
        Inverse method.
        "MNE" | "dSPM" | "sLORETA" | "eLORETA".
        (or "mne" | "dspm" | "sloreta" | "eloreta").
    lambda2 : float
        Regularization parameter.
    pick_ori : str
        Orientation to pick.
    morph : bool, str
        Morph method, e.g. fsaverage. Can be False.
    save : bool
        Save source estimate (default: False).
    """
    
    mne_dir = op.join(outdir, subject, "mne")
    mne_files = get_mne_filenames(outdir, subject)
    coreg_files = freesurfer_utils.get_coreg_filenames(outdir, subject)

    if inverse_operator is None:
        inv_op_fname = mne_files["inverse_operator"]
        inverse_operator = mne.minimum_norm.read_inverse_operator(inv_op_fname)

    if method == "mne":
        method = "MNE"

    if method == "dspm":
        method = "dSPM"

    if method == "sloreta":
        method = "sLORETA"

    if method == "eloreta":
        method = "eLORETA"
    
    log_or_print(f"*** Applying {method} surface inverse operator ***")

    if isinstance(data, mne.Epochs):
        stc = mne.minimum_norm.apply_inverse_epochs(
            data,
            inverse_operator,
            lambda2=lambda2,
            method=method,
            pick_ori=pick_ori,
        )

    else:
        stc = mne.minimum_norm.apply_inverse_raw(
            data,
            inverse_operator,
            lambda2=lambda2,
            method=method,
            pick_ori=pick_ori,
        )
    
    if morph:
        log_or_print(f"*** Morphing source estimate to {morph} ***")
        src_from = mne.read_source_spaces(coreg_files['source_space'])
        morph = morph_surface(outdir, subject, src_from, subject_to=morph)
        morph.save(op.join(mne_dir, morph), overwrite=True)
        stc = morph.apply(stc)
    
    if save:     
        log_or_print("*** Saving source estimate ***")
        if isinstance(data, mne.Epochs):
            stc.save(op.join(mne_dir, "src-epo"), overwrite=True)
        else:
            stc.save(op.join(mne_dir, "src-raw"), overwrite=True)

    return stc


def apply_inverse_operator_vol(
    outdir,
    subject,
    data,
    method,
    lambda2,
    pick_ori="pca",
    inverse_operator=None,
    transform=None,
):
    """Apply previously computed minimum norm inverse solution (volumetric).
    
    Parameters
    ----------
    outdir : str
        Output directory.
    subject : str
        Subject ID.
    data : mne.io.Raw, mne.Epochs
        Raw or Epochs object.
    inverse_operator : mne.minimum_norm.InverseOperator
        Inverse operator.
    method : str
        Inverse method.
        "MNE" | "dSPM" | "sLORETA" | "eLORETA".
        (or "mne" | "dspm" | "sloreta" | "eloreta").
    lambda2 : float
        Regularization parameter.
    pick_ori : str
        Orientation to pick.
    transform : str, optional
        Should we standardise ('ztrans') or de-mean ('demean') the voxel
        time courses? If None, no transform is applied.

    Returns
    -------
    ts : (voxels, time) array
        In native MRI space.
    """
    
    mne_dir = op.join(outdir, subject, "mne")
    mne_files = get_mne_filenames(outdir, subject)

    if inverse_operator is None:
        inv_op_fname = mne_files["inverse_operator"]
        inverse_operator = mne.minimum_norm.read_inverse_operator(inv_op_fname)

    if method == "mne":
        method = "MNE"

    if method == "dspm":
        method = "dSPM"

    if method == "sloreta":
        method = "sLORETA"

    if method == "eloreta":
        method = "eLORETA"
    
    log_or_print(f"*** Applying {method} volumetric inverse operator ***")

    if isinstance(data, mne.Epochs):
        raise ValueError("Currently volumetric MNE only supports Raw data.")

    if pick_ori.lower() != "pca":
        raise ValueError("pick_ori must be 'pca'.")

    # Estimate source activity
    stc = mne.minimum_norm.apply_inverse_raw(
        data,
        inverse_operator,
        lambda2=lambda2,
        pick_ori="vector",
        method=method,
        method_params={"eps": 1e-8, "max_iter": 100},
    )

    # Use PCA to project 3D value into maximum variance axis
    xyz_ts = stc.data
    source_nn = inverse_operator['source_nn'][2::3]
    del inverse_operator

    xyz_ts = xyz_ts - np.mean(xyz_ts, axis=-1, keepdims=True)
    eig_vals, eig_vecs = np.linalg.eig(np.matmul(xyz_ts,xyz_ts.transpose(0,2,1)))
    order = np.argsort(np.abs(eig_vals), axis=-1)
    max_power_ori = eig_vecs[np.arange(len(eig_vecs)), :, order[:, -1]]
    assert max_power_ori.shape == (xyz_ts.shape[0], xyz_ts.shape[1])

    signs = np.sign(np.sum(max_power_ori * source_nn, axis=1, keepdims=True))
    signs[signs == 0] = 1.0
    max_power_ori *= signs
    ts = np.squeeze(np.matmul(np.expand_dims(max_power_ori,1),xyz_ts),axis = 1)

    # Only keep good time points
    _, times = data.get_data(
            reject_by_annotation="omit", return_times=True, verbose=0
        )
    indices = data.time_as_index(times, use_rounding=True)
    ts = ts[..., indices]

    # Transform the voxel time courses
    if transform == "ztrans":
        log_or_print(f"Applying {transform} to voxel time courses")
        ts -= np.mean(ts, axis=0, keepdims=True)
        ts /= np.std(ts, axis=0, keepdims=True)
    elif transform == "demean":
        log_or_print(f"Applying {transform} to voxel time courses")
        ts -= np.mean(ts, axis=0, keepdims=True)

    return ts
    
    
def calc_noise_cov(data, data_cov_rank, chantypes):
    """Calculate noise covariance.
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw object.
    data_cov_rank : int
        Rank of the data covariance matrix.
    chantypes : list
        List of channel types to include.
    """
    # In MNE, the noise cov is normally obtained from empty room noise
    # recordings or from a baseline period. Here (if no noise cov is passed in)
    # we mimic what the osl_normalise_sensor_data.m function in Matlab OSL does,
    # by computing a diagonal noise cov with the variances set to the mean
    # variance of each sensor type (e.g. mag, grad, eeg.)
    log_or_print("*** Calculating noise covariance ***")
    
    data = data.pick(chantypes)
    if isinstance(data, mne.io.Raw):
        data_cov = mne.compute_raw_covariance(data, method="empirical", rank=data_cov_rank)
    else:
        data_cov = mne.compute_covariance(data, method="empirical", rank=data_cov_rank)
    
    n_channels = data_cov.data.shape[0]
    noise_cov_diag = np.zeros(n_channels)
    
    for type in chantypes:
        # Indices of this channel type
        type_raw = data.copy().pick(type, exclude="bads")
        inds = []
        for chan in type_raw.info["ch_names"]:
            inds.append(data_cov.ch_names.index(chan))

        # Mean variance of channels of this type
        variance = np.mean(np.diag(data_cov.data)[inds])
        noise_cov_diag[inds] = variance
        log_or_print(f"variance for chantype {type} is {variance}")

    bads = [b for b in data.info["bads"] if b in data_cov.ch_names]

    noise_cov = mne.Covariance(
        noise_cov_diag,
        data_cov.ch_names,
        bads,
        data.info["projs"],
        nfree=data.n_times,
    )

    return noise_cov


def morph_surface(
    subjects_dir,
    subject,
    src_from,
    subject_to="fsaverage",
    src_to=None,
    spacing=None,
):
    """Morph source space to another subject's surface.
    
    Parameters
    ----------
    subject : str
        Subject ID.
    subjects_dir : str
        Subjects directory.
    src_from : mne.SourceSpaces
        Original source space.
    src_to : str, mne.SourceSpaces
        Destination source space. can be 'fsaverage' or a source space.
    """
    
    # get source spacing from src_to

    fsaverage_coreg_filenames = freesurfer_utils.get_coreg_filenames(subjects_dir, "fsaverage")
    
    if subject_to == "fsaverage" and not op.exists(fsaverage_coreg_filenames['source_space']):
        # estimate source spacing from src_from
        if 'spacing' not in src_from.info['command_line']:
            src_to = freesurfer_utils.make_fsaverage_src(subjects_dir) # use default
        else:
            spacing = int(src_from.info['command_line'].split('spacing=')[1].split(', ')[0])
            src_to = freesurfer_utils.make_fsaverage_src(subjects_dir, spacing)
        
    morph = mne.compute_source_morph(
        src_from,
        subject_from=subject,
        subject_to=subject_to,
        src_to=src_to,
        subjects_dir=subjects_dir,

    )
    return morph
