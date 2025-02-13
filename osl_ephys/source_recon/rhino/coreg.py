"""Registration of Headshapes Including Nose in osl-ephys (RHINO).

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import warnings
import os.path as op
from pathlib import Path
from shutil import copyfile

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from mne import read_epochs, read_forward_solution
from mne.viz._3d import _sensor_shape
from mne.viz.backends.renderer import _get_renderer
from mne.transforms import write_trans, read_trans, apply_trans, _get_trans, combine_transforms, Transform, rotation, invert_transform
from mne.forward import _create_meg_coils
from mne.io import read_info, read_raw, RawArray

try:
    from mne import pick_types
except ImportError:
    # Depreciated in mne 1.6
    from mne.io.pick import pick_types

try:
    from mne._fiff.tag import _loc_to_coil_trans
except ImportError:
    # Depreciated in mne 1.6
    from mne.io import _loc_to_coil_trans

from fsl import wrappers as fsl_wrappers

import osl_ephys.source_recon.rhino.utils as rhino_utils
from osl_ephys.source_recon.rhino.surfaces import get_surfaces_filenames
from osl_ephys.utils.logger import log_or_print


def get_coreg_filenames(subjects_dir, subject):
    """Files used in coregistration by RHINO.

    Files will be in subjects_dir/subject/rhino/coreg.

    Parameters
    ----------
    subjects_dir : string
        Directory containing the subject directories.
    subject : string
        Subject directory name to put the coregistration files in.

    Returns
    -------
    filenames : dict
        A dict of files generated and used by RHINO.
    """
    rhino_files = rhino_utils.get_rhino_filenames(subjects_dir, subject)
    return rhino_files["coreg"]


def coreg(
    fif_file,
    subjects_dir,
    subject,
    use_headshape=True,
    use_nose=True,
    use_dev_ctf_t=True,
    already_coregistered=False,
    allow_smri_scaling=False,
    n_init=1,
):
    """Coregistration.

    Calculates a linear, affine transform from native sMRI space to polhemus (head) space, using headshape points that include the nose
    (if useheadshape = True). Requires ``rhino.compute_surfaces`` to have been run. This is based on the OSL Matlab version of RHINO.
    Call ``get_coreg_filenames(subjects_dir, subject)`` to get a file list of generated files. RHINO firsts registers the polhemus-derived
    fiducials (nasion, rpa, lpa) in polhemus space to the sMRI-derived fiducials in native sMRI space.

    RHINO then refines this by making use of polhemus-derived headshape points that trace out the surface of the head (scalp), and ideally include the nose.

    Finally, these polhemus-derived headshape points in polhemus space are registered to the sMRI-derived scalp surface in native sMRI space.

    In more detail:
    
    1)  Map location of fiducials in MNI standard space brain to native sMRI space. These are then used as the location of the sMRI-derived fiducials in native sMRI space.
    
    2a) We have polhemus-derived fids in polhemus space and sMRI-derived fids in native sMRI space. Use these to estimate the affine xform from native sMRI space to polhemus
        (head) space.
        
    2b) We can also optionally learn the best scaling to add to this affine xform, such that the sMRI-derived fids are scaled in size to better match the polhemus-derived fids.
        This assumes that we trust the size (e.g. in mm) of the polhemus-derived fids, but not the size of sMRI-derived fids. E.g. this might be the case if we do not trust
        the size (e.g. in mm) of the sMRI, or if we are using a template sMRI that would has not come from this subject.
        
    3)  If a scaling is learnt in step 2, we apply it to sMRI, and to anything derived from sMRI.
    
    4)  Transform sMRI-derived headshape points into polhemus space.
    
    5)  We have the polhemus-derived headshape points in polhemus space and the sMRI-derived headshape (scalp surface) in native sMRI space.  Use these to estimate the affine 
        xform from native sMRI space using the ICP algorithm initilaised using the xform estimate in step 2.

    Parameters
    ----------
    fif_file : string
        Full path to MNE-derived fif file.
    subjects_dir : string
        Directory to put RHINO subject dirs in. Files will be in subjects_dir/subject/coreg.
    subject : string
        Subject name dir to put RHINO files in. Files will be in subjects_dir/subject/coreg.
    use_headshape : bool
        Determines whether polhemus derived headshape points are used.
    use_nose : bool
        Determines whether nose is used to aid coreg, only relevant if use_headshape=True.
    use_dev_ctf_t : bool
        Determines whether to set dev_head_t equal to dev_ctf_t in fif_file's info. This option is only potentially needed for fif files originating from CTF scanners.
        Will be ignored if dev_ctf_t does not exist in info (e.g. if the data is from a MEGIN scanner)
    already_coregistered : bool
        Indicates that the data is already coregistered. Causes a simplified coreg to be run that assumes that device space, head space and mri space are all the same space,
        and that the sensor locations and polhemus points (if there are any) are already in that space. This means that dev_head_t is identity and that dev_mri_t is identity.
        This simplified coreg is needed to ensure that all the necessary coreg output files are created.
    allow_smri_scaling : bool
        Indicates if we are to allow scaling of the sMRI, such that the sMRI-derived fids are scaled in size to better match the polhemus-derived fids. This assumes that we
        trust the size (e.g. in mm) of the polhemus-derived fids, but not the size of the sMRI-derived fids. E.g. this might be the case if we do not trust the size (e.g. in mm)
        of the sMRI, or if we are using a template sMRI that has not come from this subject.
    n_init : int
        Number of initialisations for the ICP algorithm that performs coregistration.
    """

    # Note the jargon used varies for xforms and coord spaces:
    # MEG (device) -- dev_head_t --> HEAD (polhemus)
    # HEAD (polhemus)-- head_mri_t (polhemus2native) --> MRI (native)
    # MRI (native) -- mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices
    #
    # RHINO does everthing in mm

    log_or_print("*** RUNNING OSL-EPHYS RHINO COREGISTRATION ***")

    filenames = get_coreg_filenames(subjects_dir, subject)
    surfaces_filenames = get_surfaces_filenames(subjects_dir, subject)

    # -------------------------------------------------------------------------------------------------------------------
    # Copy fif_file to new file for modification, and (optionally) changes dev_head_t to equal dev_ctf_t in fif file info

    if "raw.fif" in fif_file:
        raw = read_raw(fif_file)
    elif "epo.fif" in fif_file:
        raw = read_epochs(fif_file)
    else:
        raise ValueError("Invalid fif file, needs to be a *-raw.fif or a *-epo.fif file")
    info = raw.info

    if use_dev_ctf_t:
        dev_ctf_t = raw.info["dev_ctf_t"]
        if dev_ctf_t is not None:
            log_or_print("Detected CTF data")
            log_or_print("Setting dev_head_t equal to dev_ctf_t in fif file info.")
            log_or_print("To turn this off, set use_dev_ctf_t=False")
            dev_head_t, _ = _get_trans(raw.info["dev_head_t"], "meg", "head")
            dev_head_t["trans"] = dev_ctf_t["trans"]

    raw = RawArray(np.zeros([len(info["ch_names"]), 1]), info)
    raw.save(filenames["info_fif_file"], overwrite=True)

    if already_coregistered:
        # Data is already coregistered.

        # Assumes that device space, head space and mri space are all the same space, and that the sensor locations and
        # polhemus points (if there are any) are already in that space. This means that dev_head_t is identity and that
        # dev_mri_t is identity.

        # Write native (mri) voxel index to native (mri) transform
        xform_nativeindex2scalednative = rhino_utils.get_sform(surfaces_filenames["bet_outskin_mesh_file"])["trans"]
        mrivoxel_scaledmri_t = Transform("mri_voxel", "mri", np.copy(xform_nativeindex2scalednative))
        write_trans(filenames["mrivoxel_scaledmri_t_file"], mrivoxel_scaledmri_t, overwrite=True)

        # head_mri-trans.fif for scaled MRI
        head_mri_t = Transform("head", "mri", np.identity(4))
        write_trans(filenames["head_mri_t_file"], head_mri_t, overwrite=True)
        write_trans(filenames["head_scaledmri_t_file"], head_mri_t, overwrite=True)

        # Copy meshes to coreg dir from surfaces dir
        for filename in [
            "smri_file",
            "bet_outskin_mesh_file",
            "bet_outskin_plus_nose_mesh_file",
            "bet_inskull_mesh_file",
            "bet_outskull_mesh_file",
            "bet_outskin_mesh_vtk_file",
            "bet_inskull_mesh_vtk_file",
            "bet_outskull_mesh_vtk_file",
        ]:
            copyfile(surfaces_filenames[filename], filenames[filename])
    else:
        # Run full coregistration

        if use_headshape:
            if use_nose:
                log_or_print("The MRI-derived nose is going to be used to aid coreg.")
                log_or_print("Please ensure that rhino.compute_surfaces was run with include_nose=True.")
                log_or_print("Please ensure that the polhemus headshape points include the nose.")
            else:
                log_or_print("The MRI-derived nose is not going to be used to aid coreg.")
                log_or_print("Please ensure that the polhemus headshape points do not include the nose")

        # Load in the "polhemus-derived fiducial points"
        log_or_print(f"loading: {filenames['polhemus_headshape_file']}")
        polhemus_headshape_polhemus = np.loadtxt(filenames["polhemus_headshape_file"])

        log_or_print(f"loading: {filenames['polhemus_nasion_file']}")
        polhemus_nasion_polhemus = np.loadtxt(filenames["polhemus_nasion_file"])

        log_or_print(f"loading: {filenames['polhemus_rpa_file']}")
        polhemus_rpa_polhemus = np.loadtxt(filenames["polhemus_rpa_file"])

        log_or_print(f"loading: {filenames['polhemus_lpa_file']}")
        polhemus_lpa_polhemus = np.loadtxt(filenames["polhemus_lpa_file"])

        # Load in outskin_mesh_file to get the "sMRI-derived headshape points"
        if use_nose:
            outskin_mesh_file = filenames["bet_outskin_plus_nose_mesh_file"]
        else:
            outskin_mesh_file = filenames["bet_outskin_mesh_file"]

        # ---------------------------------------------------------------------------------------------------------
        # 1) Map location of fiducials in MNI standard space brain to native sMRI space. These are then used as the
        #    location of the sMRI-derived fiducials in native sMRI space.

        if Path(filenames["mni_nasion_mni_file"]).exists() and Path(filenames["mni_rpa_mni_file"]).exists() and Path(filenames["mni_lpa_mni_file"]).exists():
            # Load recorded fiducials in MNI space
            log_or_print("Reading MNI fiducials from file")

            log_or_print(f"loading: {filenames['mni_nasion_mni_file']}")
            mni_nasion_mni = np.loadtxt(filenames["mni_nasion_mni_file"])

            log_or_print(f"loading: {filenames['mni_rpa_mni_file']}")
            mni_rpa_mni = np.loadtxt(filenames["mni_rpa_mni_file"])

            log_or_print(f"loading: {filenames['mni_lpa_mni_file']}")
            mni_lpa_mni = np.loadtxt(filenames["mni_lpa_mni_file"])

        else:
            # Known locations of MNI derived fiducials in MNI coords in mm
            log_or_print("Using known MNI fiducials")
            mni_nasion_mni = np.asarray([1, 85, -41])
            mni_rpa_mni = np.asarray([83, -20, -65])
            mni_lpa_mni = np.asarray([-83, -20, -65])

        mni_mri_t = read_trans(surfaces_filenames["mni_mri_t_file"])

        # Apply this xform to the mni fids to get what we call the "sMRI-derived fids" in native space
        smri_nasion_native = rhino_utils.xform_points(mni_mri_t["trans"], mni_nasion_mni)
        smri_lpa_native = rhino_utils.xform_points(mni_mri_t["trans"], mni_lpa_mni)
        smri_rpa_native = rhino_utils.xform_points(mni_mri_t["trans"], mni_rpa_mni)

        # ---------------------------------------------------------------------------------------------------------
        # 2a) We have polhemus-derived fids in polhemus space and sMRI-derived fids in native sMRI space. Use these
        #     to estimate the affine xform from native sMRI space to polhemus (head) space.
        #
        # 2b) We can also optionally learn the best scaling to add to this affine xform, such that the sMRI-derived
        #     fids are scaled in size to better match the polhemus-derived fids. This assumes that we trust the size
        #     (e.g. in mm) of the polhemus-derived fids, but not the size of the sMRI-derived fids. E.g. this might
        #     be the case if we do not trust the size (e.g. in mm) of the sMRI, or if we are using a template sMRI
        #     that has not come from this subject.

        # Note that smri_fid_native are the sMRI-derived fids in native space
        polhemus_fid_polhemus = np.concatenate(
            (np.reshape(polhemus_nasion_polhemus, [-1, 1]), np.reshape(polhemus_rpa_polhemus, [-1, 1]), np.reshape(polhemus_lpa_polhemus, [-1, 1])),
            axis=1,
        )
        smri_fid_native = np.concatenate((np.reshape(smri_nasion_native, [-1, 1]), np.reshape(smri_rpa_native, [-1, 1]), np.reshape(smri_lpa_native, [-1, 1])), axis=1)

        # Estimate the affine xform from native sMRI space to polhemus (head) space.
        # Optionally includes a scaling of the sMRI, captured by xform_native2scalednative
        xform_scalednative2polhemus, xform_native2scalednative = rhino_utils.rigid_transform_3D(polhemus_fid_polhemus, smri_fid_native, compute_scaling=allow_smri_scaling)

        # ---------------------------------------------------------------------------------------------------
        # 3) Apply scaling from xform_native2scalednative to sMRI, and to stuff derived from sMRI, including:
        #    - sMRI
        #    - sMRI-derived surfaces
        #    - sMRI-derived fiducials

        # Scale sMRI and sMRI-derived mesh files by changing their sform
        xform_nativeindex2native = rhino_utils.get_sform(surfaces_filenames["smri_file"])["trans"]
        xform_nativeindex2scalednative = xform_native2scalednative @ xform_nativeindex2native
        for filename in ["smri_file", "bet_outskin_mesh_file", "bet_outskin_plus_nose_mesh_file", "bet_inskull_mesh_file", "bet_outskull_mesh_file"]:
            copyfile(surfaces_filenames[filename], filenames[filename])
            # Command: fslorient -setsform <sform> <smri_file>
            sform = xform_nativeindex2scalednative.flatten()
            fsl_wrappers.misc.fslorient(filenames[filename], setsform=tuple(sform))

        # Scale vtk meshes
        for mesh_fname, vtk_fname in zip(
            ["bet_outskin_mesh_file", "bet_inskull_mesh_file", "bet_outskull_mesh_file"],
            ["bet_outskin_mesh_vtk_file", "bet_inskull_mesh_vtk_file", "bet_outskull_mesh_vtk_file"],
        ):
            rhino_utils.transform_vtk_mesh(surfaces_filenames[vtk_fname], surfaces_filenames[mesh_fname], filenames[vtk_fname], filenames[mesh_fname], xform_native2scalednative)

        # Put sMRI-derived fiducials into scaled sMRI space
        xform = xform_native2scalednative @ mni_mri_t["trans"]
        smri_nasion_scalednative = rhino_utils.xform_points(xform, mni_nasion_mni)
        smri_lpa_scalednative = rhino_utils.xform_points(xform, mni_lpa_mni)
        smri_rpa_scalednative = rhino_utils.xform_points(xform, mni_rpa_mni)

        # -----------------------------------------------------------------------
        # 4) Now we can transform sMRI-derived headshape pnts into polhemus space

        # Get native (mri) voxel index to scaled native (mri) transform
        xform_nativeindex2scalednative = rhino_utils.get_sform(outskin_mesh_file)["trans"]

        # Put sMRI-derived headshape points into native space (in mm)
        smri_headshape_nativeindex = rhino_utils.niimask2indexpointcloud(outskin_mesh_file)
        smri_headshape_scalednative = rhino_utils.xform_points(xform_nativeindex2scalednative, smri_headshape_nativeindex)

        # Put sMRI-derived headshape points into polhemus space
        smri_headshape_polhemus = rhino_utils.xform_points(xform_scalednative2polhemus, smri_headshape_scalednative)

        # ---------------------------------------------------------------------------------------------------
        # 5) We have the polhemus-derived headshape points in polhemus space and the sMRI-derived headshape
        #    (scalp surface) in native sMRI space. We use these to estimate the affine xform from native sMRI
        #    space using the ICP algorithm initilaised using the xform estimate in step 2.

        if use_headshape:
            log_or_print("Running ICP...")

            # Run ICP with multiple initialisations to refine registration of sMRI-derived headshape points to
            # polhemus derived headshape points, with both in polhemus space

            # Combined polhemus-derived headshape points and polhemus-derived fids, with them both in polhemus space
            # These are the "source" points that will be moved around
            polhemus_headshape_polhemus_4icp = np.concatenate((polhemus_headshape_polhemus, polhemus_fid_polhemus), axis=1)

            xform_icp, err, e = rhino_utils.rhino_icp(smri_headshape_polhemus, polhemus_headshape_polhemus_4icp, n_init=n_init)

        else:
            # No refinement by ICP:
            xform_icp = np.eye(4)

        # Create refined xforms using result from ICP
        xform_scalednative2polhemus_refined = np.linalg.inv(xform_icp) @ xform_scalednative2polhemus

        # Put sMRI-derived fiducials into refined polhemus space
        smri_nasion_polhemus = rhino_utils.xform_points(xform_scalednative2polhemus_refined, smri_nasion_scalednative)
        smri_rpa_polhemus = rhino_utils.xform_points(xform_scalednative2polhemus_refined, smri_rpa_scalednative)
        smri_lpa_polhemus = rhino_utils.xform_points(xform_scalednative2polhemus_refined, smri_lpa_scalednative)

        # ---------------
        # Save coreg info

        # Save xforms in MNE format in mm

        # Save xform from head to mri for the scaled mri
        xform_scalednative2polhemus_refined_copy = np.copy(xform_scalednative2polhemus_refined)
        head_scaledmri_t = Transform("head", "mri", np.linalg.inv(xform_scalednative2polhemus_refined_copy))
        write_trans(filenames["head_scaledmri_t_file"], head_scaledmri_t, overwrite=True)

        # Save xform from head to mri for the unscaled mri, this is needed if we later want to map back into MNI space
        # from head space following source recon, i.e. by combining this xform with surfaces_filenames['mni_mri_t_file']
        xform_native2polhemus_refined = np.linalg.inv(xform_icp) @ xform_scalednative2polhemus @ xform_native2scalednative
        xform_native2polhemus_refined_copy = np.copy(xform_native2polhemus_refined)
        head_mri_t = Transform("head", "mri", np.linalg.inv(xform_native2polhemus_refined_copy))
        write_trans(filenames["head_mri_t_file"], head_mri_t, overwrite=True)

        # Save xform from mrivoxel to mri
        nativeindex_scalednative_t = np.copy(xform_nativeindex2scalednative)
        mrivoxel_scaledmri_t = Transform("mri_voxel", "mri", nativeindex_scalednative_t)
        write_trans(filenames["mrivoxel_scaledmri_t_file"], mrivoxel_scaledmri_t, overwrite=True)

        # save sMRI derived fids in mm in polhemus space
        np.savetxt(filenames["smri_nasion_file"], smri_nasion_polhemus)
        np.savetxt(filenames["smri_rpa_file"], smri_rpa_polhemus)
        np.savetxt(filenames["smri_lpa_file"], smri_lpa_polhemus)

    # ---------------------------------------------------------------------------------------------
    # Create sMRI-derived freesurfer meshes in native/mri space in mm, for use by forward modelling

    nativeindex_scalednative_t = np.copy(xform_nativeindex2scalednative)
    mrivoxel_scaledmri_t = Transform("mri_voxel", "mri", nativeindex_scalednative_t)
    rhino_utils.create_freesurfer_meshes_from_bet_surfaces(filenames, mrivoxel_scaledmri_t["trans"])

    log_or_print('rhino.coreg_display("{}", "{}") can be used to check the result'.format(subjects_dir, subject))
    log_or_print("*** OSL-EPHYS RHINO COREGISTRATION COMPLETE ***")


def coreg_metrics(subjects_dir, subject):
    """Calculate metrics that summarise the coregistration.

    Parameters
    ----------
    subjects_dir : string
        Directory containing RHINO subject directories.
    subject : string
        Subject name directory containing RHINO files.

    Returns
    -------
    fiducial_distances : np.ndarray
        Distance in cm between the polhemus and sMRI fiducials. Order is nasion, lpa, rpa.
    """
    coreg_filenames = get_coreg_filenames(subjects_dir, subject)
    smri_nasion_file = coreg_filenames["smri_nasion_file"]
    smri_rpa_file = coreg_filenames["smri_rpa_file"]
    smri_lpa_file = coreg_filenames["smri_lpa_file"]
    polhemus_nasion_file = coreg_filenames["polhemus_nasion_file"]
    polhemus_rpa_file = coreg_filenames["polhemus_rpa_file"]
    polhemus_lpa_file = coreg_filenames["polhemus_lpa_file"]
    info_fif_file = coreg_filenames["info_fif_file"]

    info = read_info(info_fif_file)
    dev_head_t, _ = _get_trans(info["dev_head_t"], "meg", "head")
    dev_head_t["trans"][0:3, -1] = dev_head_t["trans"][0:3, -1] * 1000
    head_trans = invert_transform(dev_head_t)

    # Load polhemus fidcials, these are in mm
    if op.isfile(polhemus_nasion_file):
        polhemus_nasion_polhemus = np.loadtxt(polhemus_nasion_file)
        polhemus_nasion_meg = rhino_utils.xform_points(head_trans["trans"], polhemus_nasion_polhemus)

    if op.isfile(polhemus_rpa_file):
        polhemus_rpa_polhemus = np.loadtxt(polhemus_rpa_file)
        polhemus_rpa_meg = rhino_utils.xform_points(head_trans["trans"], polhemus_rpa_polhemus)

    if op.isfile(polhemus_lpa_file):
        polhemus_lpa_polhemus = np.loadtxt(polhemus_lpa_file)
        polhemus_lpa_meg = rhino_utils.xform_points(head_trans["trans"], polhemus_lpa_polhemus)

    # Load sMRI derived fids, these are in mm in polhemus/head space
    if op.isfile(smri_nasion_file):
        smri_nasion_polhemus = np.loadtxt(smri_nasion_file)
        smri_nasion_meg = rhino_utils.xform_points(head_trans["trans"], smri_nasion_polhemus)

    if op.isfile(smri_rpa_file):
        smri_rpa_polhemus = np.loadtxt(smri_rpa_file)
        smri_rpa_meg = rhino_utils.xform_points(head_trans["trans"], smri_rpa_polhemus)

    if op.isfile(smri_lpa_file):
        smri_lpa_polhemus = np.loadtxt(smri_lpa_file)
        smri_lpa_meg = rhino_utils.xform_points(head_trans["trans"], smri_lpa_polhemus)

    # Distance between polhemus and sMRI fiducials in cm
    nasion_distance = np.sqrt(np.sum((polhemus_nasion_meg - smri_nasion_meg) ** 2))
    lpa_distance = np.sqrt(np.sum((polhemus_lpa_meg - smri_lpa_meg) ** 2))
    rpa_distance = np.sqrt(np.sum((polhemus_rpa_meg - smri_rpa_meg) ** 2))
    distances = np.array([nasion_distance, lpa_distance, rpa_distance]) * 1e-1

    return distances


def coreg_display(
    subjects_dir,
    subject,
    plot_type="surf",
    display_outskin=True,
    display_outskin_with_nose=True,
    display_sensors=True,
    display_sensor_oris=True,
    display_fiducials=True,
    display_headshape_pnts=True,
    filename=None,
):
    """Display coregistration.

    Displays the coregistered RHINO scalp surface and polhemus/sensor locations.

    Display is done in MEG (device) space (in mm).

    Purple dots are the polhemus derived fiducials (these only get used to initialse the coreg, if headshape points are being used).

    Yellow diamonds are the MNI standard space derived fiducials (these are the ones that matter).

    Parameters
    ----------
    subjects_dir : string
        Directory to put RHINO subject dirs in. Files will be in subjects_dir/subject/rhino/coreg.
    subject : string
        Subject name dir to put RHINO files in. Files will be in subjects_dir/subject/rhino/coreg.
    plot_type : string
        Either:
            'surf' to do a 3D surface plot using surface meshes.
            'scatter' to do a scatter plot using just point clouds.
    display_outskin_with_nose : bool
        Whether to show nose with scalp surface in the display.
    display_outskin : bool
        Whether to show scalp surface in the display.
    display_sensors : bool
        Whether to include sensors in the display.
    display_sensor_oris - bool
        Whether to include sensor orientations in the display.
    display_fiducials - bool
        Whether to include fiducials in the display.
    display_headshape_pnts - bool
        Whether to include headshape points in the display.
    filename : str
        Filename to save display to (as an interactive html).
        Must have extension .html.
    """

    # Note the jargon used varies for xforms and coord spaces:
    # MEG (device) -- dev_head_t --> HEAD (polhemus)
    # HEAD (polhemus)-- head_mri_t (polhemus2native) --> MRI (native)
    # MRI (native) -- mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices
    #
    # RHINO does everthing in mm

    coreg_filenames = get_coreg_filenames(subjects_dir, subject)

    bet_outskin_plus_nose_mesh_file = coreg_filenames["bet_outskin_plus_nose_mesh_file"]
    bet_outskin_mesh_file = coreg_filenames["bet_outskin_mesh_file"]
    bet_outskin_mesh_vtk_file = coreg_filenames["bet_outskin_mesh_vtk_file"]
    bet_outskin_surf_file = coreg_filenames["bet_outskin_surf_file"]
    bet_outskin_plus_nose_surf_file = coreg_filenames["bet_outskin_plus_nose_surf_file"]

    head_scaledmri_t_file = coreg_filenames["head_scaledmri_t_file"]
    mrivoxel_scaledmri_t_file = coreg_filenames["mrivoxel_scaledmri_t_file"]
    smri_nasion_file = coreg_filenames["smri_nasion_file"]
    smri_rpa_file = coreg_filenames["smri_rpa_file"]
    smri_lpa_file = coreg_filenames["smri_lpa_file"]
    polhemus_nasion_file = coreg_filenames["polhemus_nasion_file"]
    polhemus_rpa_file = coreg_filenames["polhemus_rpa_file"]
    polhemus_lpa_file = coreg_filenames["polhemus_lpa_file"]
    polhemus_headshape_file = coreg_filenames["polhemus_headshape_file"]
    info_fif_file = coreg_filenames["info_fif_file"]

    if display_outskin_with_nose:
        outskin_mesh_file = bet_outskin_plus_nose_mesh_file
        outskin_mesh_4surf_file = bet_outskin_plus_nose_mesh_file
        outskin_surf_file = bet_outskin_plus_nose_surf_file
    else:
        outskin_mesh_file = bet_outskin_mesh_file
        outskin_mesh_4surf_file = bet_outskin_mesh_vtk_file
        outskin_surf_file = bet_outskin_surf_file

    # ------------
    # Setup xforms

    info = read_info(info_fif_file)

    mrivoxel_scaledmri_t = read_trans(mrivoxel_scaledmri_t_file)

    head_scaledmri_t = read_trans(head_scaledmri_t_file)
    # get meg to head xform in metres from info
    dev_head_t, _ = _get_trans(info["dev_head_t"], "meg", "head")

    # Change xform from metres to mm.
    # Note that MNE xform in fif.info assume metres, whereas we want it
    # in mm. To change units for an xform, just need to change the translation
    # part and leave the rotation alone
    dev_head_t["trans"][0:3, -1] = dev_head_t["trans"][0:3, -1] * 1000

    # We are going to display everything in MEG (device) coord frame in mm
    head_trans = invert_transform(dev_head_t)
    meg_trans = Transform("meg", "meg")
    mri_trans = invert_transform(combine_transforms(dev_head_t, head_scaledmri_t, "meg", "mri"))

    # -------------------------------
    # Setup fids and headshape points

    if display_fiducials:

        # Load polhemus derived fids, these are in mm in polhemus/head space

        polhemus_nasion_meg = None
        if op.isfile(polhemus_nasion_file):
            # Load, these are in mm
            polhemus_nasion_polhemus = np.loadtxt(polhemus_nasion_file)

            # Move to MEG (device) space
            polhemus_nasion_meg = rhino_utils.xform_points(head_trans["trans"], polhemus_nasion_polhemus)

        polhemus_rpa_meg = None
        if op.isfile(polhemus_rpa_file):
            # Load, these are in mm
            polhemus_rpa_polhemus = np.loadtxt(polhemus_rpa_file)

            # Move to MEG (device) space
            polhemus_rpa_meg = rhino_utils.xform_points(head_trans["trans"], polhemus_rpa_polhemus)

        polhemus_lpa_meg = None
        if op.isfile(polhemus_lpa_file):
            # Load, these are in mm
            polhemus_lpa_polhemus = np.loadtxt(polhemus_lpa_file)

            # Move to MEG (device) space
            polhemus_lpa_meg = rhino_utils.xform_points(head_trans["trans"], polhemus_lpa_polhemus)

        # Load sMRI derived fids, these are in mm in polhemus/head space

        smri_nasion_meg = None
        if op.isfile(smri_nasion_file):
            # Load, these are in mm
            smri_nasion_polhemus = np.loadtxt(smri_nasion_file)

            # Move to MEG (device) space
            smri_nasion_meg = rhino_utils.xform_points(head_trans["trans"], smri_nasion_polhemus)

        smri_rpa_meg = None
        if op.isfile(smri_rpa_file):
            # Load, these are in mm
            smri_rpa_polhemus = np.loadtxt(smri_rpa_file)

            # Move to MEG (device) space
            smri_rpa_meg = rhino_utils.xform_points(head_trans["trans"], smri_rpa_polhemus)

        smri_lpa_meg = None
        if op.isfile(smri_lpa_file):
            # Load, these are in mm
            smri_lpa_polhemus = np.loadtxt(smri_lpa_file)

            # Move to MEG (device) space
            smri_lpa_meg = rhino_utils.xform_points(head_trans["trans"], smri_lpa_polhemus)

    if display_headshape_pnts:
        polhemus_headshape_meg = None
        if op.isfile(polhemus_headshape_file):
            polhemus_headshape_polhemus = np.loadtxt(polhemus_headshape_file)
            polhemus_headshape_meg = rhino_utils.xform_points(head_trans["trans"], polhemus_headshape_polhemus)

    # -----------------
    # Setup MEG sensors

    if display_sensors or display_sensor_oris:

        meg_picks = pick_types(info, meg=True, ref_meg=False, exclude=())

        coil_transs = [_loc_to_coil_trans(info["chs"][pick]["loc"]) for pick in meg_picks]
        coils = _create_meg_coils([info["chs"][pick] for pick in meg_picks], acc="normal")

        meg_rrs, meg_tris, meg_sensor_locs, meg_sensor_oris = (list(), list(), list(), list())
        offset = 0
        for coil, coil_trans in zip(coils, coil_transs):
            rrs, tris = _sensor_shape(coil)
            rrs = apply_trans(coil_trans, rrs)
            meg_rrs.append(rrs)
            meg_tris.append(tris + offset)

            sens_locs = np.array([[0, 0, 0]])
            sens_locs = apply_trans(coil_trans, sens_locs)

            # MNE assumes that affine transform to determine sensor location/orientation
            # is applied to a unit vector along the z-axis
            sens_oris = np.array([[0, 0, 1]]) * 0.01
            sens_oris = apply_trans(coil_trans, sens_oris)
            sens_oris = sens_oris - sens_locs
            meg_sensor_locs.append(sens_locs)
            meg_sensor_oris.append(sens_oris)

            offset += len(meg_rrs[-1])

        if len(meg_rrs) == 0:
            log_or_print("MEG sensors not found. Cannot plot MEG locations.")
        else:
            meg_rrs = apply_trans(meg_trans, np.concatenate(meg_rrs, axis=0))
            meg_sensor_locs = apply_trans(meg_trans, np.concatenate(meg_sensor_locs, axis=0))
            meg_sensor_oris = apply_trans(meg_trans, np.concatenate(meg_sensor_oris, axis=0))
            meg_tris = np.concatenate(meg_tris, axis=0)

        # convert to mm
        meg_rrs = meg_rrs * 1000
        meg_sensor_locs = meg_sensor_locs * 1000
        meg_sensor_oris = meg_sensor_oris * 1000

    # --------
    # Do plots

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if plot_type == "surf":
            # Initialize figure
            renderer = _get_renderer(None, bgcolor=(0.5, 0.5, 0.5), size=(500, 500))

            if display_headshape_pnts:
                # Polhemus-derived headshape points
                if polhemus_headshape_meg is not None and len(polhemus_headshape_meg.T) > 0:
                    polhemus_headshape_megt = polhemus_headshape_meg.T
                    if len(polhemus_headshape_megt) < 200:
                        scale = 0.007
                    elif len(polhemus_headshape_megt) >= 200 and len(polhemus_headshape_megt) < 400:
                        scale = 0.005
                    elif len(polhemus_headshape_megt) >= 400:
                        scale = 0.003
                    color, alpha = "red", 1
                    renderer.sphere(center=polhemus_headshape_megt, color=color, scale=scale * 1000, opacity=alpha, backface_culling=True)
                else:
                    log_or_print("There are no headshape points to display")

            if display_fiducials:

                # MRI-derived nasion, rpa, lpa
                if smri_nasion_meg is not None and len(smri_nasion_meg.T) > 0:
                    color, scale, alpha = "yellow", 0.09, 1
                    for data in [smri_nasion_meg.T, smri_rpa_meg.T, smri_lpa_meg.T]:
                        transform = np.eye(4)
                        transform[:3, :3] = mri_trans["trans"][:3, :3] * scale * 1000
                        # rotate around Z axis 45 deg first
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
                    log_or_print("There are no MRI derived fiducials to display")

                # Polhemus-derived nasion, rpa, lpa
                if polhemus_nasion_meg is not None and len(polhemus_nasion_meg.T) > 0:
                    color, scale, alpha = "pink", 0.012, 1
                    for data in [polhemus_nasion_meg.T, polhemus_rpa_meg.T, polhemus_lpa_meg.T]:
                        renderer.sphere(center=data, color=color, scale=scale * 1000, opacity=alpha, backface_culling=True)
                else:
                    log_or_print("There are no polhemus derived fiducials to display")

            if display_sensors:
                # Sensors
                if len(meg_rrs) > 0:
                    color, alpha = (0.0, 0.25, 0.5), 0.2
                    surf = dict(rr=meg_rrs, tris=meg_tris)
                    renderer.surface(surface=surf, color=color, opacity=alpha, backface_culling=True)

            if display_sensor_oris:

                if len(meg_rrs) > 0:
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

            if display_outskin or display_outskin_with_nose:

                # sMRI-derived scalp surface
                # if surf file does not exist, then we must create it
                rhino_utils.create_freesurfer_mesh_from_bet_surface(
                    infile=outskin_mesh_4surf_file,
                    surf_outfile=outskin_surf_file,
                    nii_mesh_file=outskin_mesh_file,
                    xform_mri_voxel2mri=mrivoxel_scaledmri_t["trans"],
                )

                coords_native, faces = nib.freesurfer.read_geometry(outskin_surf_file)

                # Move to MEG (device) space
                coords_meg = rhino_utils.xform_points(mri_trans["trans"], coords_native.T).T

                surf_smri = dict(rr=coords_meg, tris=faces)

                renderer.surface(surface=surf_smri, color=(0, 0.7, 0.7), opacity=0.4, backface_culling=False)

            renderer.set_camera(azimuth=90, elevation=90, distance=600, focalpoint=(0.0, 0.0, 0.0))

            # Save or show
            rhino_utils.save_or_show_renderer(renderer, filename)

        # --------------------------
        elif plot_type == "scatter":

            # -------------------
            # Setup scalp surface

            # Load in scalp surface
            # And turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
            smri_headshape_nativeindex = rhino_utils.niimask2indexpointcloud(outskin_mesh_file)

            # Move from native voxel indices to native space coordinates (in mm)
            smri_headshape_native = rhino_utils.xform_points(mrivoxel_scaledmri_t["trans"], smri_headshape_nativeindex)

            # Move to MEG (device) space
            smri_headshape_meg = rhino_utils.xform_points(mri_trans["trans"], smri_headshape_native)

            plt.figure()
            ax = plt.axes(projection="3d")

            if display_sensors:
                color, scale, alpha, marker = (0.0, 0.25, 0.5), 1, 0.1, "."
                if len(meg_rrs) > 0:
                    meg_rrst = meg_rrs.T  # do plot in mm
                    ax.scatter(meg_rrst[0, :], meg_rrst[1, :], meg_rrst[2, :], color=color, marker=marker, s=scale, alpha=alpha)

            if display_sensor_oris:
                if len(meg_rrs) > 0:
                    ax.quiver(
                        meg_sensor_locs[:, 0],
                        meg_sensor_locs[:, 1],
                        meg_sensor_locs[:, 2],
                        meg_sensor_oris[:, 0],
                        meg_sensor_oris[:, 1],
                        meg_sensor_oris[:, 2],
                        arrow_length_ratio=0.3,
                        length=1.5,
                    )

            if display_outskin or display_outskin_with_nose:
                color, scale, alpha, marker = (0, 0.7, 0.7), 4, 0.3, "o"
                if len(smri_headshape_meg) > 0:
                    smri_headshape_megt = smri_headshape_meg
                    ax.scatter(smri_headshape_megt[0, 0:-1:10], smri_headshape_megt[1, 0:-1:10], smri_headshape_megt[2, 0:-1:10], color=color, marker=marker, s=scale, alpha=alpha)

            if display_headshape_pnts:
                color, scale, alpha, marker = "red", 8, 0.7, "o"
                if polhemus_headshape_meg is not None and len(polhemus_headshape_meg) > 0:
                    polhemus_headshape_megt = polhemus_headshape_meg
                    ax.scatter(polhemus_headshape_megt[0, :], polhemus_headshape_megt[1, :], polhemus_headshape_megt[2, :], color=color, marker=marker, s=scale, alpha=alpha)
                else:
                    log_or_print("There are no headshape points to plot")

            if display_fiducials:

                if smri_nasion_meg is not None and len(smri_nasion_meg) > 0:
                    color, scale, alpha, marker = (1, 1, 0), 200, 1, "d"
                    for data in (smri_nasion_meg, smri_rpa_meg, smri_lpa_meg):
                        datat = data
                        ax.scatter(datat[0, :], datat[1, :], datat[2, :], color=color, marker=marker, s=scale, alpha=alpha)
                else:
                    log_or_print("There are no structural MRI derived fiducials to plot")

                if polhemus_nasion_meg is not None and len(polhemus_nasion_meg) > 0:
                    color, scale, alpha, marker = (1, 0.5, 0.7), 400, 1, "."
                    for data in (polhemus_nasion_meg, polhemus_rpa_meg, polhemus_lpa_meg):
                        datat = data
                        ax.scatter(datat[0, :], datat[1, :], datat[2, :], color=color, marker=marker, s=scale, alpha=alpha)
                else:
                    log_or_print("There are no polhemus derived fiducials to plot")

            if filename is None:
                plt.show()
            else:
                log_or_print(f"saving {filename}")
                plt.savefig(filename)
                plt.close()
        else:
            raise ValueError("invalid plot_type.")


def bem_display(
    subjects_dir,
    subject,
    plot_type="surf",
    display_outskin_with_nose=True,
    display_sensors=False,
    filename=None,
):
    """Displays the coregistered RHINO scalp surface and inner skull surface.

    Display is done in MEG (device) space (in mm).

    Parameters
    ----------
    subjects_dir : string
        Directory to find RHINO subject dirs in.
    subject : string
        Subject name dir to find RHINO files in.
    plot_type : string
        Either:
            'surf' to do a 3D surface plot using surface meshes.
            'scatter' to do a scatter plot using just point clouds.
    display_outskin_with_nose : bool
        Whether to include nose with scalp surface in the display.
    display_sensors : bool
        Whether to include sensor locations in the display.
    filename : str
        Filename to save display to (as an interactive html). Must have extension .html.
    """

    # Note the jargon used varies for xforms and coord spaces:
    # MEG (device) -- dev_head_t --> HEAD (polhemus)
    # HEAD (polhemus)-- head_mri_t (polhemus2native) --> MRI (native)
    # MRI (native) -- mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices
    #
    # RHINO does everthing in mm

    rhino_files = rhino_utils.get_rhino_filenames(subjects_dir, subject)
    filenames = rhino_files["coreg"]

    bet_outskin_plus_nose_mesh_file = filenames["bet_outskin_plus_nose_mesh_file"]
    bet_outskin_plus_nose_surf_file = filenames["bet_outskin_plus_nose_surf_file"]
    bet_outskin_mesh_file = filenames["bet_outskin_mesh_file"]
    bet_outskin_mesh_vtk_file = filenames["bet_outskin_mesh_vtk_file"]
    bet_outskin_surf_file = filenames["bet_outskin_surf_file"]
    bet_inskull_mesh_file = filenames["bet_inskull_mesh_file"]
    bet_inskull_surf_file = filenames["bet_inskull_surf_file"]

    head_scaledmri_t_file = filenames["head_scaledmri_t_file"]
    mrivoxel_scaledmri_t_file = filenames["mrivoxel_scaledmri_t_file"]

    info_fif_file = filenames["info_fif_file"]

    if display_outskin_with_nose:
        outskin_mesh_file = bet_outskin_plus_nose_mesh_file
        outskin_mesh_4surf_file = bet_outskin_plus_nose_mesh_file
        outskin_surf_file = bet_outskin_plus_nose_surf_file
    else:
        outskin_mesh_file = bet_outskin_mesh_file
        outskin_mesh_4surf_file = bet_outskin_mesh_vtk_file
        outskin_surf_file = bet_outskin_surf_file

    fwd_fname = rhino_files["fwd_model"]
    if Path(fwd_fname).exists():
        forward = read_forward_solution(fwd_fname)
        src = forward["src"]
    else:
        src = None

    # ------------
    # Setup xforms

    info = read_info(info_fif_file)

    mrivoxel_scaledmri_t = read_trans(mrivoxel_scaledmri_t_file)

    # get meg to head xform in metres from info
    head_scaledmri_t = read_trans(head_scaledmri_t_file)
    dev_head_t, _ = _get_trans(info["dev_head_t"], "meg", "head")

    # Change xform from metres to mm.
    # Note that MNE xform in fif.info assume metres, whereas we want it
    # in mm. To change units on an xform, just need to change the translation
    # part and leave the rotation alone
    dev_head_t["trans"][0:3, -1] = dev_head_t["trans"][0:3, -1] * 1000

    # We are going to display everything in MEG (device) coord frame in mm
    meg_trans = Transform("meg", "meg")
    mri_trans = invert_transform(combine_transforms(dev_head_t, head_scaledmri_t, "meg", "mri"))
    head_trans = invert_transform(dev_head_t)

    # -----------------
    # Setup MEG sensors

    if display_sensors:
        meg_picks = pick_types(info, meg=True, ref_meg=False, exclude=())

        coil_transs = [_loc_to_coil_trans(info["chs"][pick]["loc"]) for pick in meg_picks]
        coils = _create_meg_coils([info["chs"][pick] for pick in meg_picks], acc="normal")

        meg_rrs, meg_tris = list(), list()
        offset = 0
        for coil, coil_trans in zip(coils, coil_transs):
            rrs, tris = _sensor_shape(coil)
            rrs = apply_trans(coil_trans, rrs)
            meg_rrs.append(rrs)
            meg_tris.append(tris + offset)
            offset += len(meg_rrs[-1])
        if len(meg_rrs) == 0:
            log_or_print("MEG sensors not found. Cannot plot MEG locations.")
        else:
            meg_rrs = apply_trans(meg_trans, np.concatenate(meg_rrs, axis=0))
            meg_tris = np.concatenate(meg_tris, axis=0)

        # convert to mm
        meg_rrs = meg_rrs * 1000

    # ----------------------------
    # Setup vol source grid points

    if src is not None:
        # stored points are in metres, convert to mm
        src_pnts = src[0]["rr"][src[0]["vertno"], :] * 1000

        # Move from head space to MEG (device) space
        src_pnts = rhino_utils.xform_points(head_trans["trans"], src_pnts.T).T

        log_or_print("BEM surface: number of dipoles = {}".format(src_pnts.shape[0]))

    # --------
    # Do plots

    if plot_type == "surf":
        # Initialize figure
        renderer = _get_renderer(None, bgcolor=(0.5, 0.5, 0.5), size=(500, 500))

        # Sensors
        if display_sensors:
            if len(meg_rrs) > 0:
                color, alpha = (0.0, 0.25, 0.5), 0.2
                surf = dict(rr=meg_rrs, tris=meg_tris)
                renderer.surface(surface=surf, color=color, opacity=alpha, backface_culling=True)

        # sMRI-derived scalp surface
        rhino_utils.create_freesurfer_mesh_from_bet_surface(
            infile=outskin_mesh_4surf_file,
            surf_outfile=outskin_surf_file,
            nii_mesh_file=outskin_mesh_file,
            xform_mri_voxel2mri=mrivoxel_scaledmri_t["trans"],
        )

        coords_native, faces = nib.freesurfer.read_geometry(outskin_surf_file)

        # Move to MEG (device) space
        coords_meg = rhino_utils.xform_points(mri_trans["trans"], coords_native.T).T

        surf_smri = dict(rr=coords_meg, tris=faces)

        # plot surface
        renderer.surface(surface=surf_smri, color=(0.85, 0.85, 0.85), opacity=0.3, backface_culling=False)

        # Inner skull surface
        # Load in surface, this is in mm
        coords_native, faces = nib.freesurfer.read_geometry(bet_inskull_surf_file)

        # Move to MEG (device) space
        coords_meg = rhino_utils.xform_points(mri_trans["trans"], coords_native.T).T

        surf_smri = dict(rr=coords_meg, tris=faces)

        # Plot surface
        renderer.surface(surface=surf_smri, color=(0.25, 0.25, 0.25), opacity=0.25, backface_culling=False)

        # vol source grid points
        if src is not None and len(src_pnts.T) > 0:
            color, scale, alpha = (1, 0, 0), 0.001, 1
            renderer.sphere(center=src_pnts, color=color, scale=scale * 1000, opacity=alpha, backface_culling=True)

        renderer.set_camera(azimuth=90, elevation=90, distance=600, focalpoint=(0.0, 0.0, 0.0))

        # Save or show
        rhino_utils.save_or_show_renderer(renderer, filename)

    # --------------------------
    elif plot_type == "scatter":

        # -------------------
        # Setup scalp surface

        # Load in scalp surface and turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
        smri_headshape_nativeindex = rhino_utils.niimask2indexpointcloud(outskin_mesh_file)

        # Move from native voxel indices to native space coordinates (in mm)
        smri_headshape_native = rhino_utils.xform_points(mrivoxel_scaledmri_t["trans"], smri_headshape_nativeindex)

        # Move to MEG (device) space
        smri_headshape_meg = rhino_utils.xform_points(mri_trans["trans"], smri_headshape_native)

        # -------------------------
        # Setup inner skull surface

        # Load in inner skull surface and turn the nvoxx x nvoxy x nvoxz volume into a 3 x npoints point cloud
        inner_skull_nativeindex = rhino_utils.niimask2indexpointcloud(bet_inskull_mesh_file)

        # Move from native voxel indices to native space coordinates (in mm)
        inner_skull_native = rhino_utils.xform_points(mrivoxel_scaledmri_t["trans"], inner_skull_nativeindex)

        # Move to MEG (device) space
        inner_skull_meg = rhino_utils.xform_points(mri_trans["trans"], inner_skull_native)

        ax = plt.axes(projection="3d")

        # Sensors
        if display_sensors:
            color, scale, alpha, marker = (0.0, 0.25, 0.5), 2, 0.2, "."
            if len(meg_rrs) > 0:
                meg_rrst = meg_rrs.T  # do plot in mm
                ax.scatter(meg_rrst[0, :], meg_rrst[1, :], meg_rrst[2, :], color=color, marker=marker, s=scale, alpha=alpha)

        # Scalp
        color, scale, alpha, marker = (0.75, 0.75, 0.75), 6, 0.2, "."
        if len(smri_headshape_meg) > 0:
            smri_headshape_megt = smri_headshape_meg
            ax.scatter(smri_headshape_megt[0, 0:-1:20], smri_headshape_megt[1, 0:-1:20], smri_headshape_megt[2, 0:-1:20], color=color, marker=marker, s=scale, alpha=alpha)

        # Inner skull
        inner_skull_megt = inner_skull_meg
        color, scale, alpha, marker = (0.5, 0.5, 0.5), 6, 0.2, "."
        ax.scatter(inner_skull_megt[0, 0:-1:20], inner_skull_megt[1, 0:-1:20], inner_skull_megt[2, 0:-1:20], color=color, marker=marker, s=scale, alpha=alpha)

        # vol source grid points
        if src is not None and len(src_pnts.T) > 0:
            color, scale, alpha, marker = (1, 0, 0), 1, 0.5, "."
            src_pntst = src_pnts.T
            ax.scatter(
                src_pntst[0, :],
                src_pntst[1, :],
                src_pntst[2, :],
                color=color,
                marker=marker,
                s=scale,
                alpha=alpha,
            )

        if filename is None:
            plt.show()
        else:
            log_or_print(f"saving {filename}")
            plt.savefig(filename)
            plt.close()
    else:
        raise ValueError("invalid plot_type")
