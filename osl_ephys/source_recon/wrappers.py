"""Wrappers for source reconstruction.

This module contains the functions callable using a 'source_recon'
section of a config.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>


import os
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg') 
import mne
from mne.coreg import Coregistration
from mne.io import read_info
import numpy as np

from . import (
    rhino,
    beamforming,
    parcellation,
    sign_flipping,
    minimum_norm as osle_minimum_norm,
    freesurfer_utils,
)
from ..report import src_report
from ..report.preproc_report import plot_freqbands
from ..utils.logger import log_or_print


# --------------
# RHINO wrappers


def rescale_sensor_positions(outdir, subject, rescale):
    """Wrapper to move sensor positions.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    rescale : list, optional
        List containing scaling factors for the x,y,z coordinates
        of the headshape points and fiducials: [xscale, yscale, zscale].
    """
    rhino.rescale_sensor_positions(
        fif_file=f"{outdir}/{subject}/{subject}_preproc-raw.fif",
        xscale=rescale[0],
        yscale=rescale[1],
        zscale=rescale[2],
    )


def extract_polhemus_from_info(
    outdir,
    subject,
    include_eeg_as_headshape=False,
    include_hpi_as_headshape=True,
    rescale=None,
    preproc_file=None,
    epoch_file=None,
):
    """Wrapper function to extract fiducials/headshape points.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    include_eeg_as_headshape : bool, optional
        Should we include EEG locations as headshape points?
    include_hpi_as_headshape : bool, optional
        Should we include HPI locations as headshape points?
    rescale : list, optional
        List containing scaling factors for the x,y,z coordinates
        of the headshape points and fiducials: [xscale, yscale, zscale].
    preproc_file : str, optional
        Path to the preprocessed fif file.
    epoch_file : str, optional
        Path to the preprocessed fif file.
    """
    if preproc_file is None:
        preproc_file = epoch_file
    filenames = rhino.get_coreg_filenames(outdir, subject)
    rhino.extract_polhemus_from_info(
        fif_file=preproc_file,
        headshape_outfile=filenames["polhemus_headshape_file"],
        nasion_outfile=filenames["polhemus_nasion_file"],
        rpa_outfile=filenames["polhemus_rpa_file"],
        lpa_outfile=filenames["polhemus_lpa_file"],
        include_eeg_as_headshape=include_eeg_as_headshape,
        include_hpi_as_headshape=include_hpi_as_headshape,
        rescale=rescale,
    )


def extract_fiducials_from_fif(*args, **kwargs):
    """Wrapper for extract_polhemus_from_info."""
    # Kept for backwards compatibility
    extract_polhemus_from_info(*args, **kwargs)


def remove_stray_headshape_points(outdir, subject, nose=True):
    """Remove stray headshape points.

    This function removes headshape points on the nose, neck and far from the head.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    noise : bool, optional
        Should we remove headshape points near the nose?
        Useful to remove these if we have defaced structurals or aren't
        extracting the nose from the structural.
    """
    rhino.remove_stray_headshape_points(outdir, subject, nose=nose)


def save_mni_fiducials(outdir, subject, filepath):
    """Wrapper to save MNI fiducials.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    filepath : str
        Full path to the text file containing the fiducials.

        Any reference to '{subject}' (or '{0}') is replaced by the subject ID.
        E.g. 'data/fiducials/{subject}_smri_fids.txt' with subject='sub-001'
        will become 'data/fiducials/sub-001_smri_fids.txt'.

        The file must be in MNI space with the following format:

            nas -0.5 77.5 -32.6
            lpa -74.4 -20.0 -27.2
            rpa 75.4 -21.1 -21.9

        Note, the first column (fiducial naming) is ignored but the rows must
        be in the above order, i.e. be (nasion, left, right).

        The order of the coordinates is the same as given in FSLeyes.
    """
    filenames = rhino.get_coreg_filenames(outdir, subject)
    if "{0}" in filepath:
        fiducials_file = filepath.format(subject)
    else:
        fiducials_file = filepath.format(subject=subject)
    rhino.save_mni_fiducials(
        fiducials_file=fiducials_file,
        nasion_outfile=filenames["mni_nasion_mni_file"],
        rpa_outfile=filenames["mni_rpa_mni_file"],
        lpa_outfile=filenames["mni_lpa_mni_file"],
    )


def extract_polhemus_from_pos(outdir, subject, filepath):
    """Wrapper to save polhemus data from a .pos file.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    filepath : str
        Full path to the pos file for this subject.
        Any reference to '{subject}' (or '{0}') is replaced by the subject ID.
        E.g. 'data/{subject}/meg/{subject}_headshape.pos' with subject='sub-001'
        becomes 'data/sub-001/meg/sub-001_headshape.pos'.
    """
    rhino.extract_polhemus_from_pos(outdir, subject, filepath)


def extract_polhemus_from_elc(
    outdir,
    subject,
    filepath,
    remove_headshape_near_nose=False,
):
    """Wrapper to save polhemus data from an .elc file.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    filepath : str
        Full path to the elc file for this subject.
        Any reference to '{subject}' (or '{0}') is replaced by the subject ID.
        E.g. 'data/{subject}/meg/{subject}_headshape.elc' with subject='sub-001'
        becomes 'data/sub-001/meg/sub-001_headshape.elc'.
    remove_headshape_near_nose : bool, optional
        Should we remove any headshape points near the nose?
    """
    rhino.extract_polhemus_from_elc(
        outdir, subject, filepath, remove_headshape_near_nose
    )


def compute_surfaces(
    outdir,
    subject,
    smri_file,
    include_nose=True,
    cleanup_files=True,
    recompute_surfaces=False,
    do_mri2mniaxes_xform=True,
    use_qform=False,
    reportdir=None,
):
    """Wrapper for computing surfaces.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    smri_file : str
        Path to the T1 weighted structural MRI file to use in source
        reconstruction.
    include_nose : bool, optional
        Should we include the nose when we're extracting the surfaces?
    cleanup_files : bool, optional
        Specifies whether to cleanup intermediate files in the surfaces directory.
    recompute_surfaces : bool, optional
        Specifies whether or not to run compute_surfaces, if the passed
        in options have already been run.
    do_mri2mniaxes_xform : bool, optional
        Specifies whether to do step 1) of compute_surfaces, i.e. transform
        sMRI to be aligned with the MNI axes. Sometimes needed when the sMRI
        goes out of the MNI FOV after step 1).
    use_qform : bool, optional
        Should we replace the sform with the qform? Useful if the sform code
        is incompatible with OSL, but the qform is compatible.
    reportdir : str, optional
        Path to report directory.
    """
    if smri_file == "standard":
        std_struct = "MNI152_T1_2mm.nii.gz"
        log_or_print(f"Using standard structural: {std_struct}")
        smri_file = os.path.join(os.environ["FSLDIR"], "data", "standard", std_struct)

    # Compute surfaces
    already_computed = rhino.compute_surfaces(
        smri_file=smri_file,
        subjects_dir=outdir,
        subject=subject,
        include_nose=include_nose,
        cleanup_files=cleanup_files,
        recompute_surfaces=recompute_surfaces,
        do_mri2mniaxes_xform=do_mri2mniaxes_xform,
        use_qform=use_qform,
    )

    # Plot surfaces
    surface_plots = rhino.plot_surfaces(
        outdir, subject, include_nose, already_computed
    )
    surface_plots = [s.replace(f"{outdir}/", "") for s in surface_plots]

    if reportdir is not None:
        # Save info for the report
        src_report.add_to_data(
            f"{reportdir}/{subject}/data.pkl",
            {
                "compute_surfaces": True,
                "include_nose": include_nose,
                "do_mri2mniaxes_xform": do_mri2mniaxes_xform,
                "use_qform": use_qform,
                "surface_plots": surface_plots,
            },
        )


def make_watershed_bem(
    outdir, 
    subject, 
    overwrite=False,
    reportdir=None):
    """Wrapper for making the watershed BEM.
    
    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    reportdir : str, optional
        Path to report directory.
    """
    log_or_print("*** RUNNING MNE (FREESURFER) MAKE WATERSHED BEM***")
    # Make watershed BEM
    # freesurfer_utils.make_watershed_bem(
    #     outdir=outdir,
    #     subject=subject,
    #     overwrite=overwrite,
    # )
    surfaces = ['axial', 'coronal', 'sagittal']
    surf_dir = freesurfer_utils.get_freesurfer_filenames(subjects_dir=outdir, subject=subject)['surf']['basedir'].__str__()
    output_files = [f"{surf_dir.replace(outdir.__str__() + '/', '')}/{surface}.png" for surface in surfaces]

    for surf, file in zip(surfaces, output_files):
        plot_bem_kwargs = dict(
        subject=subject,
        subjects_dir=outdir,
        brain_surfaces="orig",
        orientation=surf,
        slices=range(60,201,20),
        )
        fig = mne.viz.plot_bem(**plot_bem_kwargs)
        fig.savefig(os.path.join(outdir, file))
    
    # Save info for the report
    if reportdir is not None:
        # Save info for the report
        src_report.add_to_data(
            f"{reportdir}/{subject}/data.pkl",
            {
                "compute_surfaces": True,
                "surface_plots": output_files,
            },
        )
    

def coregister(
    outdir,
    subject,
    preproc_file=None,
    epoch_file=None,
    surface_extraction_method='fsl',
    use_nose=True,
    use_headshape=True,
    already_coregistered=False,
    allow_smri_scaling=False,
    n_init=None,
    reportdir=None,
    **kwargs,
):
    """Wrapper for coregistration.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str, optional
        Path to the preprocessed fif file.
    epoch_file : str, optional
        Path to the preprocessed epochs fif file.
    surface_extraction_method : str, optional
        Method used to extract the surfaces. Can be 'fsl' or 'freesurfer'.
    use_nose : bool, optional
        Should we use the nose in the coregistration?
    use_headshape : bool, optional
        Should we use the headshape points in the coregistration?
    already_coregistered : bool, optional
        Indicates that the data is already coregistered.
    allow_smri_scaling : bool, str,  optional
        Indicates if we are to allow scaling of the sMRI, such that
        the sMRI-derived fids are scaled in size to better match the
        polhemus-derived fids. This assumes that we trust the size
        (e.g. in mm) of the polhemus-derived fids, but not the size
        of the sMRI-derived fids. E.g. this might be the case if we
        do not trust the size (e.g. in mm) of the sMRI, or if we are
        using a template sMRI that has not come from this subject.
        if in surface_extraction_method='freesurfer', this can be 'uniform' or '3-axis'.
    n_init : int, optional
        Number of initialisations for coregistration. Different defaults 
        for surface_extraction_method='fsl' and surface_extraction_method='freesurfer'
    reportdir : str, optional
        Path to report directory.
    """
    if preproc_file is None:
        preproc_file = epoch_file

    # Run coregistration
    if surface_extraction_method == "fsl":
        if n_init is None:
            n_init = 1
        
        rhino.coreg(
            fif_file=preproc_file,
            subjects_dir=outdir,
            subject=subject,
            use_headshape=use_headshape,
            use_nose=use_nose,
            already_coregistered=already_coregistered,
            allow_smri_scaling=allow_smri_scaling,
            n_init=n_init,
        )

        # Calculate metrics
        if already_coregistered:
            fid_err = None
        else:
            fid_err = rhino.coreg_metrics(subjects_dir=outdir, subject=subject)

        # Save plots
        coreg_dir = rhino.get_coreg_filenames(outdir, subject)["basedir"]
        # The coreg display may have to be delayed until after Dask processing because of rendering
        # issues on dask workers
        rhino.coreg_display( 
            subjects_dir=outdir,
            subject=subject,
            display_outskin_with_nose=False,
            filename=f"{coreg_dir}/coreg.html",
        )
        coreg_filename = f"{coreg_dir}/coreg.html".replace(f"{outdir}/", "")
            
    elif surface_extraction_method == 'freesurfer':
        coreg_files = freesurfer_utils.get_coreg_filenames(outdir, subject)
        coreg_filename = coreg_files['coreg_html']
        
        def save_coreg_html(filename):
            fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
            print("Saving", filename)
            fig.plotter.export_html(filename)

        info = read_info(preproc_file)
        
        raw = mne.io.RawArray(np.zeros([len(info["ch_names"]), 1]), info)
        raw.save(coreg_files["info_fif_file"], overwrite=True)
        
        fiducials = "estimated"  # get fiducials from fsaverage
        coreg = Coregistration(
            info, 
            subject, 
            outdir, 
            fiducials=fiducials
        )
        
        if allow_smri_scaling is False:
            coreg.set_scale_mode(allow_smri_scaling)
        if n_init is None:
            n_init = 20
        
        fiducials_kwargs = kwargs.pop("fit_fiducials", {})
        coreg.fit_fiducials(**fiducials_kwargs)
        
        icp_kwargs = kwargs.pop("fit_icp", {})
        coreg.fit_icp(n_iterations=n_init, **icp_kwargs)
        #coreg.omit_head_shape_points(distance=1e-3)
        
        plot_kwargs = dict(
            subject=subject,
            subjects_dir=outdir,
            surfaces="head",
            dig=True,
            show_axes=True,
        )

        dists = coreg.compute_dig_mri_distances()  # in m
        log_or_print(
            f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists * 1e3):.2f} mm "
            f"/ {np.min(dists * 1e3):.2f} mm / {np.max(dists * 1e3):.2f} mm"
        )
        
        save_coreg_html(coreg_files['coreg_html'])
        mne.write_trans(coreg_files['coreg_trans'], coreg.trans, overwrite=True)
        
        # Distance between polhemus and sMRI fiducials in m, and the median distance between the scalp and headshape points.
        lpa_distance = np.sqrt(np.sum((coreg._dig_dict["lpa"] - coreg.fiducials.dig[0]["r"]) ** 2))
        nasion_distance = np.sqrt(np.sum((coreg._dig_dict["nasion"] - coreg.fiducials.dig[1]["r"]) ** 2))
        rpa_distance = np.sqrt(np.sum((coreg._dig_dict["rpa"] - coreg.fiducials.dig[2]["r"]) ** 2))
        fid_err = np.array([nasion_distance, lpa_distance, rpa_distance, np.sqrt(np.mean(dists**2))]) * 1e2 # now in cm
            
            
    if reportdir is not None:
        # Save info for the report
        src_report.add_to_data(
            f"{reportdir}/{subject}/data.pkl",
            {
                "coregister": True,
                "surface_extraction_method": surface_extraction_method,
                "use_headshape": use_headshape,
                "use_nose": use_nose,
                "already_coregistered": already_coregistered,
                "allow_smri_scaling": allow_smri_scaling,
                "n_init_coreg": n_init,
                "fid_err": fid_err,
                "coreg_plot": coreg_filename,
            },
        )
        
        if surface_extraction_method == "freesurfer":
            src_report.add_to_data(
                f"{reportdir}/{subject}/data.pkl",
                {
                    "fiducials_kwargs": fiducials_kwargs,
                    "icp_kwargs": icp_kwargs,
                }
            )


def forward_model(
    outdir,
    subject,
    smri_file=None,
    surface_extraction_method='fsl',
    gridstep=8,
    model="Single Layer",
    source_space="volumetric",
    eeg=False,
    reportdir=None,
    **kwargs,
):
    """Wrapper for computing the forward model.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    smri_file : str, optional
        Path to the T1 weighted structural MRI file to use in source
        reconstruction. Only required if using freesurfer and a volumentric source
        space.    
    surface_extraction_method : str, optional
        Method used to extract the surfaces. Can be 'fsl' or 'freesurfer'.
    gridstep : int, optional
        A grid will be constructed with the spacing given by ``gridstep``
        in mm, generating a volume source space.
    model : str, optional
        Type of forward model to use. Can be 'Single Layer' or 'Triple Layer',
        where:
        'Single Layer' use a single layer (brain/cortex)
        'Triple Layer' uses three layers (scalp, inner skull, brain/cortex)
    source_space : str, optional
        Are we using volumetric, or surface based forward model? 
        Can be 'volumetric' (or 'vol') or 'surface' (or 'surf').
        Currently, `surface_extraction_method='fsl' is only supported
        for volumetric forward models.
    eeg : bool, optional
        Are we using EEG channels in the source reconstruction?
    reportdir : str, optional
        Path to report directory.
    """
    # Compute forward model
    if surface_extraction_method == 'fsl':
        if source_space in ['surf', 'surface']:
            raise ValueError(
                "Surface based forward models are currently not supported with "
                "surface_extraction_method='fsl'."
            )
        
        rhino.forward_model(
            subjects_dir=outdir,
            subject=subject,
            model=model,
            gridstep=gridstep,
            eeg=eeg,
        )

    elif surface_extraction_method == 'freesurfer': 
        log_or_print("*** RUNNING MNE FORWARD MODEL ***")
        filenames = freesurfer_utils.get_coreg_filenames(outdir, subject)
        fwd_fname = freesurfer_utils.get_freesurfer_filenames(outdir, subject)['fwd_model']
        
        if source_space in ['volumetric', 'vol']:
            src = mne.setup_volume_source_space(
            subjects_dir=outdir,
            pos=gridstep,
            subject=subject,
            mri=smri_file,
            )
        elif source_space in ['surf', 'surface']:
            src = mne.setup_source_space(
            subjects_dir=outdir,
            spacing=gridstep,
            subject=subject,
            add_dist="patch",
            )
            
        mne.write_source_spaces(filenames['source_space'], src, overwrite=True)
        
        conductivity = kwargs.pop("conductivity", None)
        if conductivity is None:
            if model == "Single Layer":
                conductivity = (0.3,)  # for single layer
            elif model == "Triple Layer":
                conductivity = (0.3, 0.006, 0.3)  # for three layers
            
        
        ico = kwargs.pop("ico", 4)
        mindist = kwargs.pop("mindist", 0)
        
        model = mne.make_bem_model(
            subjects_dir=outdir,
            subject=subject,
            conductivity=conductivity,
            ico=ico,
        )
        
        bem = mne.make_bem_solution(model)
        
        trans = mne.read_trans(filenames['coreg_trans'])
        info = read_info(filenames['info_fif_file'])
        fwd = mne.make_forward_solution(
            info,
            trans=trans,
            src=src,
            bem=bem,
            mindist=mindist,
        )
        mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
        log_or_print("*** FINISHED SURFACE BASED FORWARD MODEL ***")
        
    if reportdir is not None:
        # Save info for the report
        src_report.add_to_data(
            f"{reportdir}/{subject}/data.pkl",
            {
                "forward_model": True,
                "surface_extraction_method": surface_extraction_method,
                "model": model,
                "gridstep": gridstep,
                "eeg": eeg,
            },
        )
        
        if surface_extraction_method == 'freesurfer':
            src_report.add_to_data(
            f"{reportdir}/{subject}/data.pkl",
            {
                "conductivity": conductivity,
                "ico": ico,
                "mindist": mindist,
            }
            )


# -------------------------------------
# Beamforming and parcellation wrappers


def beamform(
    outdir,
    subject,
    preproc_file,
    epoch_file,
    chantypes,
    rank,
    freq_range=None,
    weight_norm="nai",
    pick_ori="max-power-pre-weight-norm",
    reg=0,
    use_bilateral_pairs=None,
    bilateral_tol=None,
    bilateral_tol_midline=None,
    reportdir=None,
):
    """Wrapper function for beamforming.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    epoch_file : str
        Path to epoched preprocessed fif file.
    chantypes : str or list of str
        Channel types to use in beamforming.
    rank : dict
        Keys should be the channel types and the value should be the rank
        to use.
    freq_range : list, optional
        Lower and upper band to bandpass filter before beamforming.
        If None, no filtering is done.
    weight_norm : str, optional
        Beamformer weight normalisation.
    pick_ori : str, optional
        Orientation of the dipoles.
    reg : float, optional
        The regularization for the whitened data covariance.
    reportdir : str, optional
        Path to report directory
    """
    log_or_print("beamforming")

    if isinstance(chantypes, str):
        chantypes = [chantypes]

    # Load sensor-level data
    if epoch_file is not None:
        log_or_print("using epoched data")
        data = mne.read_epochs(epoch_file, preload=True)
    else:
        data = mne.io.read_raw_fif(preproc_file, preload=True)

    # Bandpass filter
    if freq_range is not None:
        log_or_print(f"bandpass filtering: {freq_range[0]}-{freq_range[1]} Hz")
        data = data.filter(
            l_freq=freq_range[0],
            h_freq=freq_range[1],
            method="iir",
            iir_params={"order": 5, "ftype": "butter"},
        )

    # Pick channels
    chantype_data = data.copy().pick(chantypes)

    # Create beamforming filters
    log_or_print("beamforming.make_lcmv")
    log_or_print(f"chantypes: {chantypes}")
    log_or_print(f"rank: {rank}")
    filters = beamforming.make_lcmv(
        subjects_dir=outdir,
        subject=subject,
        data=chantype_data,
        chantypes=chantypes,
        reg=reg,
        weight_norm=weight_norm,
        pick_ori=pick_ori,
        rank=rank,
        save_filters=True,
        use_bilateral_pairs=use_bilateral_pairs,
        bilateral_tol=bilateral_tol,
        bilateral_tol_midline=bilateral_tol_midline,
    )

    # Make plots
    filters_cov_plot, filters_svd_plot = beamforming.make_plots(
        outdir, subject, filters, chantype_data
    )
    filters_cov_plot = filters_cov_plot.replace(f"{outdir}/", "")
    filters_svd_plot = filters_svd_plot.replace(f"{outdir}/", "")

    dipole_locations_plot = f"{subject}/dipole_locations_plot.html"
    beamforming.plot_dipole_locations(outdir, 
                                    subject, 
                                    use_bilateral_pairs, 
                                    bilateral_tol, 
                                    bilateral_tol_midline, 
                                    f"{outdir}/{dipole_locations_plot}")
    
    if reportdir is not None:
        # Save info for the report
        src_report.add_to_data(
            f"{reportdir}/{subject}/data.pkl",
            {
                "beamform": True,
                "chantypes": chantypes,
                "rank": rank,
                "reg": reg,
                "freq_range": freq_range,
                "filters_cov_plot": filters_cov_plot,
                "filters_svd_plot": filters_svd_plot,
                "dipole_locations_plot": dipole_locations_plot,
            },
        )


def minimum_norm(
    outdir,
    subject,
    preproc_file,
    epoch_file,
    chantypes,
    rank,
    surface_extraction_method="fsl",
    depth=0.8,
    loose="auto",
    lambda2=1.0/9,
    pick_ori="pca",
    freq_range=None,
    reportdir=None,
    **kwargs,
):
    """Wrapper function for MNE source localization.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    epoch_file : str
        Path to epoched preprocessed fif file.
    chantypes : list
        List of channel types to include.
    rank : int
        Rank of the noise covariance matrix.
    surface_extraction_method : str
        Method used to extract the surfaces. Can be 'fsl' or 'freesurfer'.
    depth : float, optional
        Depth weighting.
    loose : float, optional
        Loose orientation constraint.
    lambda2 : float
        Regularization parameter for the minimum norm estimate.
        Use 1/9 for MEG, 1 for EEG.
    pick_ori : str
        Orientation of the dipoles.
    freq_range : list, optional
        Lower and upper band to bandpass filter before source estimation.
        If None, no filtering is done.
    reportdir : str, optional
        Path to report directory.
    """
    os.environ["SUBJECTS_DIR"] = str(outdir)

    if isinstance(chantypes, str):
        chantypes = [chantypes]
    
    if epoch_file is not None:
        data = mne.read_epochs(epoch_file, preload=True)
    else:
        data = mne.io.read_raw(preproc_file, preload=True)
    
    # Bandpass filter
    if freq_range is not None:
        log_or_print(f"bandpass filtering: {freq_range[0]}-{freq_range[1]} Hz")
        data = data.filter(
            l_freq=freq_range[0],
            h_freq=freq_range[1],
            method="iir",
            iir_params={"order": 5, "ftype": "butter"},
        )

    # Create inverse operator
    if surface_extraction_method == "freesurfer":
        fs_files = freesurfer_utils.get_freesurfer_filenames(outdir, subject)
        fwd_fname = fs_files["fwd_model"]
    elif surface_extraction_method == 'fsl':
        rhino_files = rhino.utils.get_rhino_filenames(outdir, subject)
        fwd_fname = rhino_files["fwd_model"]

    mne_files = osle_minimum_norm.get_mne_filenames(outdir, subject)
    inv_op_fname = mne_files["inverse_operator"]
    
    log_or_print("creating MNE inverse operator")
    osle_minimum_norm.create_inverse_operator(
        fwd_fname,
        data,
        chantypes,
        rank,
        depth=depth,
        loose=loose,
        filename=inv_op_fname,
    )
    
    if reportdir is not None:
        # Save info for the report
        src_report.add_to_data(
            f"{reportdir}/{subject}/data.pkl",
            {
                "minimum_norm": True,
                "chantypes": chantypes,
                "rank": rank,
                "depth": depth,
                "loose": loose,
                "lambda2": lambda2,
                "pick_ori": pick_ori,
                "freq_range": freq_range,
            },
        )


def parcellate(
    outdir,
    subject,
    preproc_file,
    epoch_file,
    parcellation_file,
    method,
    orthogonalisation,
    source_space,
    surface_extraction_method='fsl',
    source_method='lcmv',
    spatial_resolution=None,
    reference_brain=None,
    voxel_trans="ztrans",
    extra_chans="stim",
    neighbour_distance=None,
    reportdir=None,
    **kwargs,
):
    """Wrapper function for parcellation.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    epoch_file : str
        Path to epoched preprocessed fif file.
    parcellation_file : str
        Path to the parcellation file to use.
    method : str
        Method to use in the parcellation.
    orthogonalisation : str, None
        Options are 'symmetric', 'local', None. If 'local', neighbour_distance 
        must be specified.
    source_space : str
        Source model to use. Can be 'volumetric' (/'vol') or 'surface' (/'surf').
    surface_extraction_method : str
        Method used to extract the surfaces. Can be 'fsl' or 'freesurfer'.
    source_method : str, optional
        Method used for source reconstruction. Can be 'lcmv' or one of the MNE methods
        ('mne', 'dspm', 'sloreta', 'eloreta').
    spatial_resolution : int, optional
        Resolution for beamforming to use for the reference brain in mm
        (must be an integer, or will be cast to nearest int). If None, then
        the gridstep used in coreg_filenames['forward_model_file'] is used.
    reference_brain : str, optional
        Default depends on surface_extraction_method (fsl or freesurfer) and source_method (lcmv or mne).
        If surface_extraction_method='fsl', defaults to 'mni'. Alternatives: 'mri' or 'unscaled_mri'.
        If surface_extraction_method='freesurfer', defaults to 'fsaverage'. Alternatives: 'mri'.
    voxel_trans : str, optional
        Should we standardise ('ztrans') or de-mean ('demean') the voxel
        time courses? If None, no normalisation is applied.
    extra_chans : str or list of str, optional
        Extra channels to include in the parc-raw.fif file.
        Defaults to 'stim'. Stim channels are always added to parc-raw.fif
        in addition to extra_chans.
    neighbour_distance : float, optional
        Distance in mm between parcel centers to consider neighbours 
        for orthogonalisation='local'.
    reportdir : str, optional
        Path to report directory.
    """
    log_or_print("parcellate")

    # Load sensor-level data
    if epoch_file is not None:
        log_or_print("using epoched data")
        data = mne.read_epochs(epoch_file, preload=True)
    else:
        data = mne.io.read_raw_fif(preproc_file, preload=True)

    if reportdir is None:
        raise ValueError(
            "This function can only be used when a report was generated "
            "when using source estimation (beamforming/minimum_norm). "
            "Please use beamform_and_parcellate or minimum_norm_and_parcellate."
        )

    available_source_methods = ["lcmv", "beamform", "mne", "dspm", "sloreta", "eloreta"]
    if source_method.lower() not in available_source_methods:
        raise ValueError(f"source_method must be one of {available_source_methods}")
    elif source_method.lower() in ["lcmv", "beamform"] and source_space in ['surf', 'surface']:
        raise ValueError(
            "Surface based source models are not supported with "
            "source_method='lcmv' or 'beamform'."
        )

    # Get settings passed to the beamform/minimum_norm wrapper
    report_data = pickle.load(open(f"{reportdir}/{subject}/data.pkl", "rb"))
    freq_range = report_data.pop("freq_range")
    chantypes = report_data.pop("chantypes")
    if isinstance(chantypes, str):
        chantypes = [chantypes]
        
    # Bandpass filter
    if freq_range is not None:
        log_or_print(f"bandpass filtering: {freq_range[0]}-{freq_range[1]} Hz")
        data = data.filter(
            l_freq=freq_range[0],
            h_freq=freq_range[1],
            method="iir",
            iir_params={"order": 5, "ftype": "butter"},
        )

    log_or_print(f"using source_method: {source_method}")

    # source recon is applied in place
    if source_method in ["lcmv", "beamform"]:
        if reference_brain is None:
            reference_brain = 'mni'
        log_or_print(f"using reference_brain: {reference_brain}")

        # Pick channels
        chantype_data = data.copy().pick(chantypes)
    
        # Load beamforming filter and apply
        filters = beamforming.load_lcmv(outdir, subject)
        bf_data = beamforming.apply_lcmv(chantype_data, filters)

        if epoch_file is not None:
            bf_data = np.transpose([bf.data for bf in bf_data], axes=[1, 2, 0])
        else:
            bf_data = bf_data.data

        bf_data_mni, _, coords_mni, _ = beamforming.transform_recon_timeseries(
            subjects_dir=outdir,
            subject=subject,
            recon_timeseries=bf_data,
            spatial_resolution=spatial_resolution,
            reference_brain=reference_brain,
        )

        # Parcellation
        log_or_print(f"using file {parcellation_file}")
        parcel_data, _, _ = parcellation.vol_parcellate_timeseries(
            parcellation_file,
            voxel_timeseries=bf_data_mni,
            voxel_coords=coords_mni,
            method=method,
            working_dir=f"{outdir}/{subject}/parc",
        )
        
    elif source_method.lower() in ["mne", "eloreta", "dspm", "sloreta", "eloreta"]:

        os.environ["SUBJECTS_DIR"] = str(outdir)

        pick_ori = report_data.pop("pick_ori")
        lambda2 = report_data.pop("lambda2")

        # surface parcellation
        if source_space in ["surf", "surface"]:

            if reference_brain is None:
                reference_brain = 'fsaverage'
            if reference_brain == 'mri':
                reference_brain = subject
            log_or_print(f"using reference_brain: {reference_brain}")

            # sources are not estimated yet; so first read in inverse solution
            stc = osle_minimum_norm.apply_inverse_operator_surf(
                outdir,
                subject,
                data=data,
                method=source_method,
                lambda2=lambda2, 
                pick_ori=pick_ori,
                inverse_operator=None,
                morph=reference_brain,
                save=False,
            )
            parcel_data = parcellation.surf_parcellate_timeseries(
                subject_dir=outdir,
                subject=reference_brain,
                stc=stc,
                method=method,
                parcellation=parcellation_file,
            )

        # volumetric parcellation
        elif source_space in ["volumetric", "vol"]:

            if reference_brain is None:
                reference_brain = "mni"
            log_or_print(f"using reference_brain: {reference_brain}")

            mne_data = osle_minimum_norm.apply_inverse_operator_vol(
                outdir,
                subject,
                data, 
                method=source_method,
                lambda2=lambda2,
                pick_ori="pca",
                inverse_operator=None,
                transform=voxel_trans,
            )

            mne_data_mni, _, coords_mni, _ = beamforming.transform_recon_timeseries(
                subjects_dir=outdir,
                subject=subject,
                recon_timeseries=mne_data,
                spatial_resolution=spatial_resolution,
                reference_brain=reference_brain,
            )

            log_or_print(f"using file {parcellation_file}")
            parcel_data, _, _ = parcellation.vol_parcellate_timeseries(
                parcellation_file,
                voxel_timeseries=mne_data_mni,
                voxel_coords=coords_mni,
                method=method,
                working_dir=f"{outdir}/{subject}/parc",
            )

    # Orthogonalisation
    if orthogonalisation not in [None, "symmetric", "local", "none", "None"]:
        raise NotImplementedError(orthogonalisation)

    if orthogonalisation == "symmetric":
        log_or_print(f"{orthogonalisation} orthogonalisation")
        parcel_data = parcellation.symmetric_orthogonalise(
            parcel_data, maintain_magnitudes=True
        )
    elif orthogonalisation == "local":
        log_or_print(f"{orthogonalisation} orthogonalisation")
        parcel_data = parcellation.local_orthogonalise(
            parcel_data, parcellation_file, neighbour_distance,
        )

    os.makedirs(f"{outdir}/{subject}/parc", exist_ok=True)
    if epoch_file is None:
        # Save parcellated data as a MNE Raw object
        parc_fif_file = f"{outdir}/{subject}/parc/{source_method}-parc-raw.fif"
        log_or_print(f"saving {parc_fif_file}")
        parc_raw = parcellation.convert2mne_raw(
            parcel_data, data, extra_chans=extra_chans
        )
        parc_raw.save(parc_fif_file, overwrite=True)
    else:
        # Save parcellated data as a MNE Epochs object
        parc_fif_file = f"{outdir}/{subject}/parc/{source_method}-parc-epo.fif"
        log_or_print(f"saving {parc_fif_file}")
        parc_epo = parcellation.convert2mne_epochs(parcel_data, data)
        parc_epo.save(parc_fif_file, overwrite=True)

    # Save plots
    parc_psd_plot = f"{subject}/parc/psd.png"
    parcellation.plot_psd(
        parcel_data,
        fs=data.info["sfreq"],
        freq_range=freq_range,
        parcellation_file=parcellation_file,
        filename=f"{outdir}/{parc_psd_plot}",
        freesurfer=surface_extraction_method=='freesurfer',
    )
    parc_corr_plot = f"{subject}/parc/corr.png"
    parcellation.plot_correlation(parcel_data, filename=f"{outdir}/{parc_corr_plot}")

    if surface_extraction_method == 'fsl':
        parc_freqbands_plot = f"{subject}/parc/freqbands.png"
        plot_freqbands(parc_raw, f"{outdir}/{parc_freqbands_plot}")
    else:
        parc_freqbands_plot = None
    
    # Save info for the report
    n_parcels = parcel_data.shape[0]
    n_samples = parcel_data.shape[1]
    if parcel_data.ndim == 3:
        n_epochs = parcel_data.shape[2]
    else:
        n_epochs = None
    src_report.add_to_data(
        f"{reportdir}/{subject}/data.pkl",
        {
            "parcellate": True,
            "parcellation_file": parcellation_file,
            "method": method,
            "reference_brain": reference_brain,
            "orthogonalisation": orthogonalisation,
            "parc_fif_file": str(parc_fif_file),
            "n_samples": n_samples,
            "n_parcels": n_parcels,
            "n_epochs": n_epochs,
            "parc_psd_plot": parc_psd_plot,
            "parc_corr_plot": parc_corr_plot,
            "parc_freqbands_plot": parc_freqbands_plot,
        },
    )


def beamform_and_parcellate(
    outdir,
    subject,
    preproc_file,
    epoch_file,
    chantypes,
    rank,
    parcellation_file,
    method,
    orthogonalisation,
    freq_range=None,
    weight_norm="nai",
    pick_ori="max-power-pre-weight-norm",
    reg=0,
    spatial_resolution=None,
    reference_brain="mni",
    extra_chans="stim",
    neighbour_distance=None,
    use_bilateral_pairs=None,
    bilateral_tol=None,
    bilateral_tol_midline=None,    
    reportdir=None,
):
    """Wrapper function for beamforming and parcellation.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    epoch_file : str
        Path to epoched preprocessed fif file.
    chantypes : str or list of str
        Channel types to use in beamforming.
    rank : dict
        Keys should be the channel types and the value should be the rank
        to use.
    parcellation_file : str
        Path to the parcellation file to use.
    method : str
        Method to use in the parcellation.
    orthogonalisation : str, None
        Options are 'symmetric', 'local', None. If 'local', neighbour_distance 
        must be specified.
    freq_range : list, optional
        Lower and upper band to bandpass filter before beamforming.
        If None, no filtering is done.
    weight_norm : str, optional
        Beamformer weight normalisation.
    pick_ori : str, optional
        Orientation of the dipoles.
    reg : float, optional
        The regularization for the whitened data covariance.
    spatial_resolution : int, optional
        Resolution for beamforming to use for the reference brain in mm
        (must be an integer, or will be cast to nearest int). If None,
        then the gridstep used in coreg_filenames['forward_model_file']
        is used.
    reference_brain : str, optional
        'mni' indicates that the reference_brain is the stdbrain in MNI space.
        'mri' indicates that the reference_brain is the subject's sMRI in the
        scaled native/mri space.
        'unscaled_mri' indicates that the reference_brain is the subject's
        sMRI in unscaled native/mri space.
        Note that Scaled/unscaled relates to the allow_smri_scaling option
        in coreg. If allow_scaling was False, then the unscaled MRI will be
        the same as the scaled MRI.
    extra_chans : str or list of str, optional
        Extra channels to include in the parc-raw.fif file.
        Defaults to 'stim'. Stim channels are always added to parc-raw.fif
        in addition to extra_chans.
    neighbour_distance : float, optional
        Distance in mm between parcel centers to consider neighbours 
        for orthogonalisation='local'.
    reportdir : str, optional
        Path to report directory.
    """
    log_or_print("beamform_and_parcellate")

    if isinstance(chantypes, str):
        chantypes = [chantypes]

    # Load sensor-level data
    if epoch_file is not None:
        log_or_print("using epoched data")
        data = mne.read_epochs(epoch_file, preload=True)
    else:
        data = mne.io.read_raw_fif(preproc_file, preload=True)

    # Bandpass filter
    if freq_range is not None:
        log_or_print(f"bandpass filtering: {freq_range[0]}-{freq_range[1]} Hz")
        data = data.filter(
            l_freq=freq_range[0],
            h_freq=freq_range[1],
            method="iir",
            iir_params={"order": 5, "ftype": "butter"},
        )

    # Pick channels
    chantype_data = data.copy().pick(chantypes)

    # Create beamforming filters
    log_or_print("beamforming.make_lcmv")
    log_or_print(f"chantypes: {chantypes}")
    log_or_print(f"rank: {rank}")
    filters = beamforming.make_lcmv(
        subjects_dir=outdir,
        subject=subject,
        data=chantype_data,
        chantypes=chantypes,
        reg=reg,
        weight_norm=weight_norm,
        pick_ori=pick_ori,
        rank=rank,
        save_filters=True,
        use_bilateral_pairs=use_bilateral_pairs,
        bilateral_tol=bilateral_tol,
        bilateral_tol_midline=bilateral_tol_midline,        
    )

    # Make plots
    filters_cov_plot, filters_svd_plot = beamforming.make_plots(
        outdir, subject, filters, chantype_data
    )
    filters_cov_plot = filters_cov_plot.replace(f"{outdir}/", "")
    filters_svd_plot = filters_svd_plot.replace(f"{outdir}/", "")

    dipole_locations_plot = f"{subject}/dipole_locations_plot.html"
    beamforming.plot_dipole_locations(outdir, 
                                    subject, 
                                    use_bilateral_pairs, 
                                    bilateral_tol, 
                                    bilateral_tol_midline, 
                                    f"{outdir}/{dipole_locations_plot}")

    # Apply beamforming
    bf_data = beamforming.apply_lcmv(chantype_data, filters)

    if epoch_file is not None:
        bf_data = np.transpose([bf.data for bf in bf_data], axes=[1, 2, 0])
    else:
        bf_data = bf_data.data

    bf_data_mni, _, coords_mni, _ = beamforming.transform_recon_timeseries(
        subjects_dir=outdir,
        subject=subject,
        recon_timeseries=bf_data,
        spatial_resolution=spatial_resolution,
        reference_brain=reference_brain,
    )

    # Parcellation
    log_or_print("parcellation")
    log_or_print(f"using file {parcellation_file}")
    parcel_data, _, _ = parcellation.vol_parcellate_timeseries(
        parcellation_file,
        voxel_timeseries=bf_data_mni,
        voxel_coords=coords_mni,
        method=method,
        working_dir=f"{outdir}/{subject}/parc",
    )

    # Orthogonalisation
    if orthogonalisation not in [None, "symmetric", "local", "none", "None"]:
        raise NotImplementedError(orthogonalisation)

    if orthogonalisation == "symmetric":
        log_or_print(f"{orthogonalisation} orthogonalisation")
        parcel_data = parcellation.symmetric_orthogonalise(
            parcel_data, maintain_magnitudes=True
        )
    elif orthogonalisation == "local":
        log_or_print(f"{orthogonalisation} orthogonalisation")
        parcel_data = parcellation.local_orthogonalise(
            parcel_data, parcellation_file, neighbour_distance,
        )

    if epoch_file is None:
        # Save parcellated data as a MNE Raw object
        parc_fif_file = f"{outdir}/{subject}/parc/lcmv-parc-raw.fif"
        log_or_print(f"saving {parc_fif_file}")
        parc_raw = parcellation.convert2mne_raw(
            parcel_data, data, extra_chans=extra_chans
        )
        parc_raw.save(parc_fif_file, overwrite=True)
    else:
        # Save parcellated data as a MNE Epochs object
        parc_fif_file = f"{outdir}/{subject}/parc/lcmv-parc-epo.fif"
        log_or_print(f"saving {parc_fif_file}")
        parc_epo = parcellation.convert2mne_epochs(parcel_data, data)
        parc_epo.save(parc_fif_file, overwrite=True)

    # Save plots
    parc_psd_plot = f"{subject}/parc/psd.png"
    parcellation.plot_psd(
        parcel_data,
        fs=data.info["sfreq"],
        freq_range=freq_range,
        parcellation_file=parcellation_file,
        filename=f"{outdir}/{parc_psd_plot}",
        freesurfer=False,
    )
    parc_corr_plot = f"{subject}/parc/corr.png"
    parcellation.plot_correlation(parcel_data, filename=f"{outdir}/{parc_corr_plot}")

    parc_freqbands_plot = f"{subject}/parc/freqbands.png"
    plot_freqbands(parc_raw, f"{outdir}/{parc_freqbands_plot}")

    if reportdir is not None:
        # Save info for the report
        n_parcels = parcel_data.shape[0]
        n_samples = parcel_data.shape[1]
        if parcel_data.ndim == 3:
            n_epochs = parcel_data.shape[2]
        else:
            n_epochs = None
        src_report.add_to_data(
            f"{reportdir}/{subject}/data.pkl",
            {
                "beamform_and_parcellate": True,
                "beamform": True,
                "parcellate": True,
                "chantypes": chantypes,
                "rank": rank,
                "reg": reg,
                "freq_range": freq_range,
                "filters_cov_plot": filters_cov_plot,
                "filters_svd_plot": filters_svd_plot,
                "dipole_locations_plot": dipole_locations_plot,
                "parcellation_file": parcellation_file,
                "method": method,
                "reference_brain": reference_brain,
                "orthogonalisation": orthogonalisation,
                "parc_fif_file": str(parc_fif_file),
                "n_samples": n_samples,
                "n_parcels": n_parcels,
                "n_epochs": n_epochs,
                "parc_psd_plot": parc_psd_plot,
                "parc_corr_plot": parc_corr_plot,
                "parc_freqbands_plot": parc_freqbands_plot,
            },
        )

def minimum_norm_and_parcellate(
    outdir,
    subject,
    preproc_file,
    epoch_file,
    source_method,
    source_space,
    chantypes,
    rank,
    method,
    parcellation_file,
    orthogonalisation,
    surface_extraction_method="fsl",
    depth=0.8,
    loose="auto",
    lambda2=1.0/9,
    pick_ori="pca",
    freq_range=None,
    spatial_resolution=None,
    reference_brain=None,
    voxel_trans="ztrans",
    extra_chans="stim",
    neighbour_distance=None,
    reportdir=None,
):
    """Wrapper function for minimum_norm and parcellation.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    epoch_file : str
        Path to epoched preprocessed fif file.
    source_method : str
        Method to use for inverse modelling (e.g., MNE, eLORETA).
    source_space : str
        Source model to use (e.g., volumetric, surface).
    chantypes : list
        List of channel types to include.
    rank : int
        Rank of the noise covariance matrix.
    method : str
        Method to use in the parcellation.
    parcellation_file : str
        Path to the parcellation file to use.
    orthogonalisation : str, None
        Options are 'symmetric', 'local', None. If 'local', neighbour_distance 
        must be specified.
    surface_extraction_method : str, optional
        Method used for surface extraction. Can be 'freesurfer' or 'fsl'.
    depth : float, optional
        Depth weighting.
    loose : float, optional
        Loose orientation constraint.
    lambda2 : float
        Regularization parameter for the minimum norm estimate.
        Use 1/9 for MEG, 1 for EEG.
    pick_ori : str
        Orientation of the dipoles.
    freq_range : list, optional
        Lower and upper band to bandpass filter before beamforming.
        If None, no filtering is done.
    spatial_resolution : int, optional
        Resolution for beamforming to use for the reference brain in mm
        (must be an integer, or will be cast to nearest int). If None, then
        the gridstep used in coreg_filenames['forward_model_file'] is used.
    reference_brain : str, optional
        Default depends on surface_extraction_method.
        If surface_extraction_method='fsl', defaults to 'mni'. Alternatives: 'mri' or 'unscaled_mri'.
        If surface_extraction_method='freesurfer', defaults to 'fsaverage'. Alternatives: 'mri'.
    voxel_trans : str, optional
        Should we standardise ('ztrans') or de-mean ('demean') the voxel
        time courses? If None, no normalisation is applied.
    extra_chans : str or list of str, optional
        Extra channels to include in the parc-raw.fif file.
        Defaults to 'stim'. Stim channels are always added to parc-raw.fif
        in addition to extra_chans.
    neighbour_distance : float, optional
        Distance in mm between parcel centers to consider neighbours 
        for orthogonalisation='local'.
    reportdir : str, optional
        Path to report directory.
    """
    log_or_print("minimum_norm_and_parcellate")
    os.environ["SUBJECTS_DIR"] = str(outdir)

    if isinstance(chantypes, str):
        chantypes = [chantypes]
    
    # Load sensor-level data
    if epoch_file is not None:
        log_or_print("using epoched data")
        data = mne.read_epochs(epoch_file, preload=True)
    else:
        data = mne.io.read_raw_fif(preproc_file, preload=True)

    # Bandpass filter
    if freq_range is not None:
        log_or_print(f"bandpass filtering: {freq_range[0]}-{freq_range[1]} Hz")
        data = data.filter(
            l_freq=freq_range[0],
            h_freq=freq_range[1],
            method="iir",
            iir_params={"order": 5, "ftype": "butter"},
        )

    # Create inverse operator
    if surface_extraction_method == "freesurfer":
        fs_files = freesurfer_utils.get_freesurfer_filenames(outdir, subject)
        fwd_fname = fs_files["fwd_model"]
    elif surface_extraction_method == 'fsl':
        rhino_files = rhino.utils.get_rhino_filenames(outdir, subject)
        fwd_fname = rhino_files["fwd_model"]

    mne_files = osle_minimum_norm.get_mne_filenames(outdir, subject)
    inv_op_fname = mne_files["inverse_operator"]
    
    log_or_print("creating MNE inverse operator")
    osle_minimum_norm.create_inverse_operator(
        fwd_fname,
        data,
        chantypes,
        rank,
        depth=depth,
        loose=loose,
        filename=inv_op_fname,
    )

    log_or_print(f"using source_method: {source_method}")

    # MNE surface parcellation
    if source_space in ["surf", "surface"]:

        if reference_brain is None or reference_brain=='None':
            reference_brain = 'fsaverage'
        if reference_brain == 'mri':
            reference_brain = subject
        log_or_print(f"using reference_brain: {reference_brain}")

        # sources are not estimated yet; so first read in inverse solution
        stc = osle_minimum_norm.apply_inverse_operator_surf(
            outdir,
            subject,
            data=data,
            method=source_method,
            lambda2=lambda2, 
            pick_ori=pick_ori,
            inverse_operator=None,
            morph=reference_brain,
            save=False,
        )
        parcel_data = parcellation.surf_parcellate_timeseries(
            subject_dir=outdir,
            subject=reference_brain,
            stc=stc,
            method=method,
            parcellation_file=parcellation_file,
        )

    # volumetric source model
    elif source_space in ["vol", "volumetric"]:

        if reference_brain is None:
            reference_brain = "mni"
        log_or_print(f"using reference_brain: {reference_brain}")

        mne_data = osle_minimum_norm.apply_inverse_operator_vol(
            outdir,
            subject,
            data, 
            method=source_method,
            lambda2=lambda2,
            pick_ori="pca",
            inverse_operator=None,
            transform=voxel_trans,
        )

        mne_data_mni, _, coords_mni, _ = beamforming.transform_recon_timeseries( 
            subjects_dir=outdir,
            subject=subject,
            recon_timeseries=mne_data,
            spatial_resolution=spatial_resolution,
            reference_brain=reference_brain,
        )

        log_or_print(f"using file {parcellation_file}")
        parcel_data, _, _ = parcellation.vol_parcellate_timeseries(
            parcellation_file,
            voxel_timeseries=mne_data_mni,
            voxel_coords=coords_mni,
            method=method,
            working_dir=f"{outdir}/{subject}/parc",
        )

    # Orthogonalisation
    if orthogonalisation not in [None, "symmetric", "local", "none", "None"]:
        raise NotImplementedError(orthogonalisation)

    if orthogonalisation == "symmetric":
        log_or_print(f"{orthogonalisation} orthogonalisation")
        parcel_data = parcellation.symmetric_orthogonalise(
            parcel_data, maintain_magnitudes=True
        )
    elif orthogonalisation == "local":
        log_or_print(f"{orthogonalisation} orthogonalisation")
        parcel_data = parcellation.local_orthogonalise(
            parcel_data, parcellation_file, neighbour_distance,
        )

    os.makedirs(f"{outdir}/{subject}/parc", exist_ok=True)
    if epoch_file is None:
        # Save parcellated data as a MNE Raw object
        parc_fif_file = f"{outdir}/{subject}/parc/{source_method.lower()}-parc-raw.fif"
        log_or_print(f"saving {parc_fif_file}")
        parc_raw = parcellation.convert2mne_raw(
            parcel_data, data, extra_chans=extra_chans
        )
        parc_raw.save(parc_fif_file, overwrite=True)
    else:
        # Save parcellated data as a MNE Epochs object
        parc_fif_file = f"{outdir}/{subject}/parc/{source_method.lower()}-parc-epo.fif"
        log_or_print(f"saving {parc_fif_file}")
        parc_epo = parcellation.convert2mne_epochs(parcel_data, data)
        parc_epo.save(parc_fif_file, overwrite=True)

    # Save plots
    parc_psd_plot = f"{subject}/parc/psd.png"
    parcellation.plot_psd(
        parcel_data,
        fs=data.info["sfreq"],
        freq_range=freq_range,
        parcellation_file=parcellation_file,
        filename=f"{outdir}/{parc_psd_plot}",
        freesurfer=surface_extraction_method=='freesurfer',
    )
    parc_corr_plot = f"{subject}/parc/corr.png"
    parcellation.plot_correlation(parcel_data, filename=f"{outdir}/{parc_corr_plot}")

    if surface_extraction_method == 'fsl':
        parc_freqbands_plot = f"{subject}/parc/freqbands.png"
        plot_freqbands(parc_raw, f"{outdir}/{parc_freqbands_plot}")
    else:
        parc_freqbands_plot = None
    
    if reportdir is not None:
        # Save info for the report
        n_parcels = parcel_data.shape[0]
        n_samples = parcel_data.shape[1]
        if parcel_data.ndim == 3:
            n_epochs = parcel_data.shape[2]
        else:
            n_epochs = None
        src_report.add_to_data(
            f"{reportdir}/{subject}/data.pkl",
            {
                "minimum_norm_and_parcellate": True,
                "minimum_norm": True,
                "parcellate": True,
                "surface_extraction_method": surface_extraction_method,
                "source_space": source_space,
                "chantypes": chantypes,
                "reference_brain": reference_brain,
                "method": source_method,
                "rank": rank,
                "depth": depth,
                "loose": loose,
                "lambda2": lambda2,
                "pick_ori": pick_ori,
                "freq_range": freq_range,
                "parcellation_file": parcellation_file,
                "method": method,
                "orthogonalisation": orthogonalisation,
                "parc_fif_file": str(parc_fif_file),
                "n_samples": n_samples,
                "n_parcels": n_parcels,
                "n_epochs": n_epochs,
                "parc_psd_plot": parc_psd_plot,
                "parc_corr_plot": parc_corr_plot,
                "parc_freqbands_plot": parc_freqbands_plot,
            },
        )

# ----------------------
# Sign flipping wrappers


def find_template_subject(
    outdir,
    subjects,
    n_embeddings=1,
    standardize=True,
    epoched=False,
    source_method="lcmv",
):
    """Function to find a good subject to align other subjects to in the sign flipping.

    Note, this function expects parcellated data to exist in the following
    location: outdir/*/parc/parc-*.fif, the * here represents subject
    directories or 'raw' vs 'epo'.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subjects : str
        Subjects to include.
    n_embeddings : int, optional
        Number of time-delay embeddings that we will use (if we are doing any).
    standardize : bool, optional
        Should we standardize (z-transform) the data before sign flipping?
    epoched : bool, optional
        Are we performing sign flipping on parc-raw.fif (epoched=False) or
        parc-epo.fif files (epoched=True)?
    source_method : str, optional
        Method to used for inverse modelling (e.g., LCMV, MNE, eLORETA).

    Returns
    -------
    template : str
        Template subject.
    """
    log_or_print("Finding template subject:")

    # Get the parcellated data files
    parc_files = []
    for subject in subjects:
        if epoched:
            parc_file = f"{outdir}/{subject}/parc/{source_method}-parc-epo.fif"
        else:
            parc_file = f"{outdir}/{subject}/parc/{source_method}-parc-raw.fif"
        if Path(parc_file).exists():
            parc_files.append(parc_file)
        else:
            log_or_print(f"Warning: {parc_file} not found")

    # Validation
    n_parc_files = len(parc_files)
    if n_parc_files < 2:
        raise ValueError(
            "two or more parcellated data files are needed to "
            f"perform sign flipping, got {n_parc_files}"
        )

    # Calculate the covariance matrix of each subject
    covs = sign_flipping.load_covariances(parc_files, n_embeddings, standardize)

    # Find a subject to use as a template
    template_index = sign_flipping.find_template_subject(covs, n_embeddings)
    template_subject = parc_files[template_index].split("/")[-3]
    log_or_print("Template for sign flipping:", template_subject)

    return template_subject


def fix_sign_ambiguity(
    outdir,
    subject,
    preproc_file,
    template,
    n_embeddings,
    standardize,
    n_init,
    n_iter,
    max_flips,
    epoched=False,
    source_method="lcmv",
    reportdir=None,
):
    """Wrapper function for fixing the dipole sign ambiguity.

    Parameters
    ----------
    outdir : str
        Path to where to output the source reconstruction files.
    subject : str
        Subject name/id.
    preproc_file : str
        Path to the preprocessed fif file.
    template : str
        Template subject.
    n_embeddings : int
        Number of time-delay embeddings that we will use (if we are doing any).
    standardize : bool
        Should we standardize (z-transform) the data before sign flipping?
    n_init : int
        Number of initializations.
    n_iter : int
        Number of sign flipping iterations per subject to perform.
    max_flips : int
        Maximum number of channels to flip in an iteration.
    epoched : bool, optional
        Are we performing sign flipping on parc-raw.fif (epoched=False) or
        parc-epo.fif files (epoched=True)?
    source_method : str, optional
        Method to used for inverse modelling (e.g., LCMV, MNE, eLORETA).
    reportdir : str, optional
        Path to report directory.
    """
    log_or_print("fix_sign_ambiguity")
    log_or_print(f"using template: {template}")

    # Get path to the parcellated data file for this subject and the template
    parc_files = []
    for sub in [subject, template]:
        if epoched:
            parc_file = f"{outdir}/{sub}/parc/{source_method}-parc-epo.fif"
        else:
            parc_file = f"{outdir}/{sub}/parc/{source_method}-parc-raw.fif"
        if not Path(parc_file).exists():
            raise ValueError(f"{parc_file} not found")
        parc_files.append(parc_file)

    # Calculate the covariance of this subject and the template
    [cov, template_cov] = sign_flipping.load_covariances(
        parc_files, n_embeddings, standardize, use_tqdm=False
    )

    # Find the channels to flip
    flips, metrics = sign_flipping.find_flips(
        cov,
        template_cov,
        n_embeddings,
        n_init,
        n_iter,
        max_flips,
        use_tqdm=False,
    )

    # Apply flips to the parcellated data
    sign_flipping.apply_flips(outdir, subject, flips, epoched=epoched, source_method=source_method)

    if reportdir is not None:
        # Save info for the report
        src_report.add_to_data(
            f"{reportdir}/{subject}/data.pkl",
            {
                "fix_sign_ambiguity": True,
                "template": template,
                "n_embeddings": n_embeddings,
                "standardize": standardize,
                "n_init": n_init,
                "n_iter": n_iter,
                "max_flips": max_flips,
                "metrics": metrics,
            },
        )


# --------------
# Other wrappers


def extract_rhino_files(outdir, subject, old_outdir):
    """Wrapper function for extracting RHINO files from a previous run.

    Parameters
    ----------
    outdir : str
        Path to the NEW source reconstruction directory.
    subject : str
        Subject name/id.
    old_outdir : str
        OLD source reconstruction directory to copy RHINO files to.
    """
    rhino.utils.extract_rhino_files(
        old_outdir, outdir, subjects=subject, gen_report=False
    )