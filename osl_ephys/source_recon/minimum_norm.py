"""Minimum norm source localization using MNE-Python
"""
    
# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>


import os
import os.path as op
import subprocess
import pickle
from pathlib import Path

import mne
from mne.coreg import Coregistration
from mne.io import read_info
import numpy as np

from . import rhino, beamforming, parcellation, sign_flipping, freesurfer_utils
from ..report import src_report
from ..utils.logger import log_or_print

import logging

logger = logging.getLogger(__name__)


def minimum_norm(
    outdir,
    subject,
    preproc_file,
    epoch_file,
    chantypes,
    method,
    rank,
    morph="fsaverage",
    lambda2=0.1,
    depth=0.8,
    loose='auto',
    # freq_range=None,
    # weight_norm="nai",
    pick_ori="normal",
    # reg=0,
    reportdir=None,
    ):
    """Run minimum norm source localization.
    
    Parameters
    ----------
    outdir : str
        Output directory.
    subject : str
        Subject ID.
    preproc_file : str  
        Preprocessed file.
    epoch_file : str
        Epoch file.
    chantypes : list
        List of channel types to include.
    method : str
        Inverse method.
    rank : int
        Rank of the data covariance matrix.
    morph : bool, str
        Morph method, e.g. fsaverage. Can be False.
    lambda2 : float
        Regularization parameter.
    depth : float
        Depth weighting.
    loose : float
        Loose parameter.
    freq_range : list
        Band pass filter applied before source estimation.
    weight_norm : str
        Weight normalization.
    pick_ori : str
        Orientation to pick.
    reg : float
        Regularization parameter.
    reportdir : str
        Report directory.
        
        """
    
    if preproc_file is None:
        preproc_file = epoch_file

    log_or_print("*** RUNNING MNE SOURCE LOCALIZATION ***")
    
    fwd_fname = freesurfer_utils.get_freesurfer_files(outdir, subject)['fwd_model']
    coreg_files = freesurfer_utils.get_coreg_filenames(outdir, subject)
    
    raw = mne.io.read_raw(preproc_file, preload=True)
    noise_cov = calc_noise_cov(raw, rank, chantypes)
    
    fwd = mne.read_forward_solution(fwd_fname)
    inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov, rank=rank, loose=loose, depth=depth)
    del fwd
    
    log_or_print(f"*** Applying {method} inverse solution ***")
    stc_raw = mne.minimum_norm.apply_inverse_raw(raw, inverse_operator, lambda2=lambda2, method=method, pick_ori=pick_ori)
    
    if epoch_file is not None:
        epochs = mne.read_epochs(epoch_file, preload=True)
        stc_epochs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse_operator, lambda2=lambda2, method=method, pick_ori=pick_ori)    
        
    if morph:
        src_from = mne.read_source_spaces(coreg_files['source_space'])
        morph = morph_surface(outdir, subject, src_from, src_to=morph)
        stc_raw = morph.apply(stc_raw)
        
        if epoch_file is not None:
            stc_epochs = morph.apply(stc_epochs)
            
    stc_raw.save(op.join(outdir, f"{subject}_src-raw"), overwrite=True)
    
    if epoch_file is not None:
        stc_epochs.save(op.join(outdir, f"{subject}_src-epo"), overwrite=True)

    
    
def calc_noise_cov(raw, data_cov_rank, chantypes):
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
    
    raw = raw.pick(chantypes)
    data_cov = mne.compute_raw_covariance(
        raw, method="empirical", rank=data_cov_rank
    )
    n_channels = data_cov.data.shape[0]
    noise_cov_diag = np.zeros(n_channels)
    
    for type in chantypes:
        # Indices of this channel type
        type_raw = raw.copy().pick(type, exclude="bads")
        inds = []
        for chan in type_raw.info["ch_names"]:
            inds.append(data_cov.ch_names.index(chan))

        # Mean variance of channels of this type
        variance = np.mean(np.diag(data_cov.data)[inds])
        noise_cov_diag[inds] = variance
        print(f"variance for chantype {type} is {variance}")

    bads = [b for b in raw.info["bads"] if b in data_cov.ch_names]
    noise_cov = mne.Covariance(
        noise_cov_diag, data_cov.ch_names, bads, raw.info["projs"], nfree=1e10
    )
    return noise_cov


def morph_surface(subjects_dir, subject, src_from, src_to="fsaverage", ico=None):
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
    
    if src_to == "fsaverage":
        # estimate source spacing from src_from
        if 'ico' not in src_from.info['command_line']:
            src_to = freesurfer_utils.make_fsaverage_src(subjects_dir) # use default
        else:
            ico = int(src_from.info['command_line'].split('ico=')[1].split(', ')[0])
            src_to = freesurfer_utils.make_fsaverage_src(subjects_dir, ico)
        
    morph = mne.compute_source_morph(
        src_from,
        subject_from=subject,
        subject_to=src_to,
        subjects_dir=subjects_dir,

    )
    return morph