"""Calculation of surfaces in RHINO.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>

import os.path as op
import warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import cv2
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg') 
import nilearn as nil
from scipy.ndimage import morphology
from sklearn.mixture import GaussianMixture
from mne.transforms import write_trans, Transform
from fsl import wrappers as fsl_wrappers

import osl_ephys.source_recon.rhino.utils as rhino_utils
from osl_ephys.utils.logger import log_or_print


def get_surfaces_filenames(subjects_dir, subject):
    """Generates a dict of files generated and used by rhino.compute_surfaces.

    Files will be in subjects_dir/subject/rhino/surfaces.

    Parameters
    ----------
    subjects_dir : string
        Directory containing the subject directories.
    subject : string
        Subject directory name to put the surfaces in.

    Returns
    -------
    filenames : dict
        A dict of files generated and used by rhino.compute_surfaces. Note that  due to the unusal naming conventions used by BET:
        - bet_inskull_*_file is actually the brain surface
        - bet_outskull_*_file is actually the inner skull surface
        - bet_outskin_*_file is the outer skin/scalp surface
    """
    rhino_files = rhino_utils.get_rhino_filenames(subjects_dir, subject)
    return rhino_files["surf"]


def check_if_already_computed(subjects_dir, subject, include_nose):
    """Checks if surfaces have already been computed.

    Parameters
    ----------
    subjects_dir : string
        Directory to put RHINO subject directories in. Files will be in subjects_dir/subject/surfaces.
    subject : string
        Subject name directory to put RHINO files in. Files will be in subjects_dir/subject/surfaces.
    include_nose : bool
        Specifies whether to add the nose to the outer skin (scalp) surface.

    Returns
    -------
    already_computed : bool
        Flag indicating if surfaces have been computed.
    """
    filenames = get_surfaces_filenames(subjects_dir, subject)
    if Path(filenames["completed"]).exists():
        with open(filenames["completed"], "r") as file:
            lines = file.readlines()
            completed_mri_file = lines[1].split(":")[1].strip()
            completed_include_nose = lines[2].split(":")[1].strip() == "True"
            is_same_mri = completed_mri_file == filenames["smri_file"]
            is_same_include_nose = completed_include_nose == include_nose
            if is_same_mri and is_same_include_nose:
                return True
    return False


def compute_surfaces(
    smri_file,
    subjects_dir,
    subject,
    include_nose=True,
    cleanup_files=True,
    recompute_surfaces=False,
    do_mri2mniaxes_xform=True,
    use_qform=False,
):
    """Compute surfaces.

    Extracts inner skull, outer skin (scalp) and brain surfaces from passed in smri_file, which is assumed to be a T1, using FSL. Assumes that the sMRI file has a valid sform.

    Call get_surfaces_filenames(subjects_dir, subject) to get a file list of generated files.

    In more detail:
    1) Transform sMRI to be aligned with the MNI axes so that BET works well
    2) Use bet to skull strip sMRI so that flirt works well
    3) Use flirt to register skull stripped sMRI to MNI space
    4) Use BET/BETSURF to get:
    a) The scalp surface (excluding nose), this gives the sMRI-derived headshape points in native sMRI space, which can be used in the headshape points registration later.
    b) The scalp surface (outer skin), inner skull and brain surface, these can be used for forward modelling later. Note that  due to the unusal naming conventions used by BET:
       - bet_inskull_mesh_file is actually the brain surface
       - bet_outskull_mesh_file is actually the inner skull surface
       - bet_outskin_mesh_file is the outer skin/scalp surface
    5) Refine scalp outline, adding nose to scalp surface (optional)
    6) Output the transform from sMRI space to MNI
    7) Output surfaces in sMRI space

    Parameters
    ----------
    smri_file : str
        Full path to structural MRI in niftii format (with .nii.gz extension). This is assumed to have a valid sform, i.e. the sform code needs to be 4 or 1, and the sform
        should transform from voxel indices to voxel coords in mm. The axis sform used to do this will be the native/sMRI axis used throughout rhino. The qform will be ignored.
    subjects_dir : str
        Directory to put RHINO subject directories in. Files will be in subjects_dir/subject/surfaces.
    subject : str
        Subject name directory to put RHINO files in. Files will be in subjects_dir/subject/surfaces.
    include_nose : bool, optional
        Specifies whether to add the nose to the outer skin (scalp) surface. This can help rhino's coreg to work better, assuming that there are headshape points that also
        include the nose. Requires the smri_file to have a FOV that includes the nose!
    cleanup_files : bool, optional
        Specifies whether to cleanup intermediate files in the surfaces directory.
    recompute_surfaces : bool, optional
        Specifies whether or not to run compute_surfaces if the passed in options have already been run.
    do_mri2mniaxes_xform : bool, optional
        Specifies whether to do step 1) above, i.e. transform sMRI to be aligned with the MNI axes. Sometimes needed when the sMRI goes out of the MNI FOV after step 1).
    use_qform : bool, optional
        Should we replace the sform with the qform? Useful if the sform code is incompatible with osl-ephys, but the qform is compatible.

    Returns
    -------
    already_computed : bool
        Flag indicating if we're using previously computed surfaces.
    """

    # Note the jargon used varies for xforms and coord spaces, e.g.:
    # MEG (device) -- dev_head_t --> HEAD (polhemus)
    # HEAD (polhemus)-- head_mri_t (polhemus2native) --> MRI (native)
    # MRI (native) -- mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices
    # MRI (native) -- sform (mri2mniaxes) --> MNI axes
    #
    # RHINO does everthing in mm

    filenames = get_surfaces_filenames(subjects_dir, subject)

    if not recompute_surfaces:
        # Check if surfaces have already been computed
        if check_if_already_computed(subjects_dir, subject, include_nose):
            log_or_print("*** OSL-EPHYS RHINO: USING PREVIOUSLY COMPUTED SURFACES ***")
            log_or_print(f"Surfaces directory: {filenames['basedir']}")
            log_or_print(f"include_nose={include_nose}")
            return True

    log_or_print("*** RUNNING OSL-EPHYS RHINO COMPUTE SURFACES ***")
    if include_nose:
        log_or_print("The nose is going to be added to the outer skin (scalp) surface.")
        log_or_print("Please ensure that the structural MRI has a FOV that includes the nose")
    else:
        log_or_print("The nose is not going to be added to the outer skin (scalp) surface")

    # Check smri_file
    smri_ext = "".join(Path(smri_file).suffixes)
    if smri_ext not in [".nii", ".nii.gz"]:
        raise ValueError("smri_file needs to be a niftii file with a .nii or .nii.gz extension")

    # Copy sMRI to new file for modification
    img = nib.load(smri_file)
    nib.save(img, filenames["smri_file"])

    # RHINO will always use the sform, and so we will set the qform to be same as sform for sMRI,
    # to stop the original qform from being used by mistake (e.g. by flirt)
    if use_qform:
        log_or_print("Using qform in surface extraction")

        # Command: fslorient -copyqform2sform <smri_file>
        fsl_wrappers.misc.fslorient(filenames['smri_file'], copyqform2sform=True)
    else:
        # Command: fslorient -copysform2qform <smri_file>
        fsl_wrappers.misc.fslorient(filenames['smri_file'], copysform2qform=True)

    # We will assume orientation of standard brain is RADIOLOGICAL. But let's check that is the case:
    std_orient = rhino_utils.get_orient(filenames["std_brain"])
    if std_orient != "RADIOLOGICAL":
        raise ValueError("Orientation of standard brain must be RADIOLOGICAL, please check output of:\n fslorient -orient {}".format(filenames["std_brain"]))

    # We will assume orientation of sMRI brain is RADIOLOGICAL. But let's check that is the case:
    smri_orient = rhino_utils.get_orient(filenames["smri_file"])

    if smri_orient != "RADIOLOGICAL" and smri_orient != "NEUROLOGICAL":
        raise ValueError("Cannot determine orientation of subject brain, please check output of:\n fslorient -getorient {}".format(filenames["smri_file"]))

    # If orientation is not RADIOLOGICAL then force it to be RADIOLOGICAL
    if smri_orient != "RADIOLOGICAL":
        log_or_print("reorienting subject brain to be RADIOLOGICAL")

        # Command: fslorient -forceradiological <smri_file>
        fsl_wrappers.misc.fslorient(filenames["smri_file"], forceradiological=True)

    log_or_print("You can use the following call to check the passed in structural MRI is appropriate,")
    log_or_print("including checking that the L-R, S-I, A-P labels are sensible:")
    log_or_print("In Python:")
    log_or_print('fsleyes("{}", "{}")'.format(filenames["smri_file"], filenames["std_brain"]))
    log_or_print("From the cmd line:")
    log_or_print("fsleyes {} {}".format(filenames["smri_file"], filenames["std_brain"]))

    # ------------------------------------------------------------------------
    # 1) Transform sMRI to be aligned with the MNI axes so that BET works well

    img = nib.load(filenames["smri_file"])
    img_density = np.sum(img.get_fdata()) / np.prod(img.get_fdata().shape)

    # We will start by transforming sMRI so that its voxel indices axes are aligned to MNI's. This helps BET work.

    # Calculate mri2mniaxes
    if do_mri2mniaxes_xform:
        flirt_mri2mniaxes_xform = rhino_utils.get_flirt_xform_between_axes(filenames["smri_file"], filenames["std_brain"])
    else:
        flirt_mri2mniaxes_xform = np.eye(4)

    # Write xform to disk so flirt can use it
    flirt_mri2mniaxes_xform_file = op.join(filenames["basedir"], "flirt_mri2mniaxes_xform.txt")
    np.savetxt(flirt_mri2mniaxes_xform_file, flirt_mri2mniaxes_xform)

    # Apply mri2mniaxes xform to smri to get smri_mniaxes, which means sMRIs voxel indices axes are aligned to be the same as MNI's
    # Command: flirt -in <smri_file> -ref <std_brain> -applyxfm -init <mri2mniaxes_xform_file> -out <smri_mni_axes_file>
    flirt_smri_mniaxes_file = op.join(filenames["basedir"], "flirt_smri_mniaxes.nii.gz")
    fsl_wrappers.flirt(filenames["smri_file"], filenames["std_brain"], applyxfm=True, init=flirt_mri2mniaxes_xform_file, out=flirt_smri_mniaxes_file)

    img = nib.load(flirt_smri_mniaxes_file)
    img_latest_density = np.sum(img.get_fdata()) / np.prod(img.get_fdata().shape)

    if 5 * img_latest_density < img_density:
        raise Exception(
            "Something is wrong with the passed in structural MRI:\n   {}\nEither it is empty or the sformcode is incorrectly set.\n"
            "Try running the following from a cmd line: \n   fsleyes {} {} \nAnd see if the standard space brain is shown in the same postcode as the structural.\n"
            "If it is not, then the sformcode in the structural image needs setting correctly.\n".format(filenames["smri_file"], filenames["std_brain"], filenames["smri_file"])
        )

    # -------------------------------------------------------
    # 2) Use BET to skull strip sMRI so that flirt works well

    # Check sMRI doesn't contain nans (this can cause segmentation faults with FSL's bet)
    if rhino_utils.check_nii_for_nan(filenames["smri_file"]):
        log_or_print("WARNING: nan found in sMRI file. This might cause issues with BET.")
        old_smri_file = Path(smri_file)
        new_smri_file = old_smri_file.with_name(old_smri_file.stem + '_fixed' + old_smri_file.suffix)
        log_or_print("If you encounter an error, it might be possible to fix the file with:")
        log_or_print(f"fslmaths {old_smri_file} -nan {new_smri_file}")

    log_or_print("Running BET pre-FLIRT...")

    # Command: bet <flirt_smri_mniaxes_file> <flirt_smri_mniaxes_bet_file>
    flirt_smri_mniaxes_bet_file = op.join(filenames["basedir"], "flirt_smri_mniaxes_bet")
    fsl_wrappers.bet(flirt_smri_mniaxes_file, flirt_smri_mniaxes_bet_file)

    # ---------------------------------------------------------
    # 3) Use flirt to register skull stripped sMRI to MNI space

    log_or_print("Running FLIRT...")

    # Flirt is run on the skull stripped brains to register the smri_mniaxes to the MNI standard brain
    #
    # Command: flirt -in <flirt_smri_mniaxes_bet_file> -ref <std_brain> -omat <flirt_mniaxes2mni_file> -o <flirt_smri_mni_bet_file>
    flirt_mniaxes2mni_file = op.join(filenames["basedir"], "flirt_mniaxes2mni.txt")
    flirt_smri_mni_bet_file = op.join(filenames["basedir"], "flirt_smri_mni_bet.nii.gz")
    fsl_wrappers.flirt(flirt_smri_mniaxes_bet_file, filenames["std_brain"], omat=flirt_mniaxes2mni_file, o=flirt_smri_mni_bet_file)

    # Calculate overall flirt transform, from MRI to MNI
    #
    # Command: convert_xfm -omat <flirt_mri2mni_xform_file> -concat <flirt_mniaxes2mni_file> <flirt_mri2mniaxes_xform_file>
    flirt_mri2mni_xform_file = op.join(filenames["basedir"], "flirt_mri2mni_xform.txt")
    fsl_wrappers.concatxfm(flirt_mri2mniaxes_xform_file, flirt_mniaxes2mni_file, flirt_mri2mni_xform_file)  # Note, the wrapper reverses the order of arguments

    # and also calculate its inverse, from MNI to MRI
    #
    # Command: convert_xfm -omat <mni2mri_flirt_xform_file>  -inverse <flirt_mri2mni_xform_file>
    mni2mri_flirt_xform_file = filenames["mni2mri_flirt_xform_file"]
    fsl_wrappers.invxfm(flirt_mri2mni_xform_file, mni2mri_flirt_xform_file)  # Note, the wrapper reverses the order of arguments

    # Move full sMRI into MNI space to do full bet and betsurf
    #
    # Command: flirt -in <smri_file> -ref <std_brain> -applyxfm -init <flirt_mri2mni_xform_file> -out <flirt_smri_mni_file>
    flirt_smri_mni_file = op.join(filenames["basedir"], "flirt_smri_mni.nii.gz")
    fsl_wrappers.flirt(filenames["smri_file"], filenames["std_brain"], applyxfm=True, init=flirt_mri2mni_xform_file, out=flirt_smri_mni_file)

    # --------------------------
    # 4) Use BET/BETSURF to get:
    # a) The scalp surface (excluding nose), this gives the sMRI-derived headshape points in native sMRI space, which can be used in the headshape points registration later.
    # b) The scalp surface (outer skin), inner skull and brain surface, these can be used for forward modelling later. Note that due to the unusal naming conventions used by BET:
    #    - bet_inskull_mesh_file is actually the brain surface
    #    - bet_outskull_mesh_file is actually the inner skull surface
    #    - bet_outskin_mesh_file is the outer skin/scalp surface

    log_or_print("Running BET and BETSURF...")

    # Run BET and BETSURF on smri to get the surface mesh (in MNI space)
    #
    # Command: bet <flirt_smri_mni_file> <flirt_smri_mni_bet_file> -A
    flirt_smri_mni_bet_file = op.join(filenames["basedir"], "flirt")
    fsl_wrappers.bet(flirt_smri_mni_file, flirt_smri_mni_bet_file, A=True)

    # ----------------------------------------------------------------
    # 5) Refine scalp outline, adding nose to scalp surface (optional)

    log_or_print("Refining scalp surface...")

    # We do this in MNI big FOV space, to allow the full nose to be included

    # Calculate flirt_mni2mnibigfov_xform
    mni2mnibigfov_xform = rhino_utils.get_flirt_xform_between_axes(from_nii=flirt_smri_mni_file, target_nii=filenames["std_brain_bigfov"])
    flirt_mni2mnibigfov_xform_file = op.join(filenames["basedir"], "flirt_mni2mnibigfov_xform.txt")
    np.savetxt(flirt_mni2mnibigfov_xform_file, mni2mnibigfov_xform)

    # Calculate overall transform, from smri to MNI big fov
    #
    # Command: convert_xfm -omat <flirt_mri2mnibigfov_xform_file> -concat <flirt_mni2mnibigfov_xform_file> <flirt_mri2mni_xform_file>"
    flirt_mri2mnibigfov_xform_file = op.join(filenames["basedir"], "flirt_mri2mnibigfov_xform.txt")
    fsl_wrappers.concatxfm(flirt_mri2mni_xform_file, flirt_mni2mnibigfov_xform_file, flirt_mri2mnibigfov_xform_file)  # Note, the wrapper reverses the order of arguments

    # Move MRI to MNI big FOV space and load in
    #
    # Command: flirt -in <smri_file> -ref <std_brain_bigfov> -applyxfm -init <flirt_mri2mnibigfov_xform_file> -out <flirt_smri_mni_bigfov_file>
    flirt_smri_mni_bigfov_file = op.join(filenames["basedir"], "flirt_smri_mni_bigfov")
    fsl_wrappers.flirt(filenames["smri_file"], filenames["std_brain_bigfov"], applyxfm=True, init=flirt_mri2mnibigfov_xform_file, out=flirt_smri_mni_bigfov_file)

    # Move scalp to MNI big FOV space and load in
    #
    # Command: flirt -in <flirt_outskin_file> -ref <std_brain_bigfov> -applyxfm -init <flirt_mni2mnibigfov_xform_file> -out <flirt_outskin_bigfov_file>
    flirt_outskin_file = op.join(filenames["basedir"], "flirt_outskin_mesh")
    flirt_outskin_bigfov_file = op.join(filenames["basedir"], "flirt_outskin_mesh_bigfov")
    fsl_wrappers.flirt(flirt_outskin_file, filenames["std_brain_bigfov"], applyxfm=True, init=flirt_mni2mnibigfov_xform_file, out=flirt_outskin_bigfov_file)
    scalp = nib.load(flirt_outskin_bigfov_file + ".nii.gz")

    # Create mask by filling outline

    # Add a border of ones to the mask, in case the complete head is not in the FOV, without this binary_fill_holes will not work
    mask = np.ones(np.add(scalp.shape, 2))

    # Note that z=100 is where the standard MNI FOV starts in the big FOV
    mask[1:-1, 1:-1, 102:-1] = scalp.get_fdata()[:, :, 101:]
    mask[:, :, :101] = 0

    # We assume that the top of the head is not cutoff by the FOV, we need to assume this so that binary_fill_holes works:
    mask[:, :, -1] = 0
    mask = morphology.binary_fill_holes(mask)

    # Remove added border
    mask[:, :, :102] = 0
    mask = mask[1:-1, 1:-1, 1:-1]

    if include_nose:
        log_or_print("Adding nose to scalp surface...")

        # Reclassify bright voxels outside of mask (to put nose inside the mask since bet will have excluded it)
        vol = nib.load(flirt_smri_mni_bigfov_file + ".nii.gz")
        vol_data = vol.get_fdata()

        # Normalise vol data
        vol_data = vol_data / np.max(vol_data.flatten())

        # Estimate observation model params of 2 class GMM with diagonal cov matrix where the two classes correspond to inside and outside the bet mask
        means = np.zeros([2, 1])
        means[0] = np.mean(vol_data[np.where(mask == 0)])
        means[1] = np.mean(vol_data[np.where(mask == 1)])
        precisions = np.zeros([2, 1])
        precisions[0] = 1 / np.var(vol_data[np.where(mask == 0)])
        precisions[1] = 1 / np.var(vol_data[np.where(mask == 1)])
        weights = np.zeros([2])
        weights[0] = np.sum((mask == 0))
        weights[1] = np.sum((mask == 1))

        # Create GMM with those observation models
        gm = GaussianMixture(n_components=2, random_state=0, covariance_type="diag")
        gm.means_ = means
        gm.precisions_ = precisions
        gm.precisions_cholesky_ = np.sqrt(precisions)
        gm.weights_ = weights

        # Classify voxels outside BET mask with GMM
        labels = gm.predict(vol_data[np.where(mask == 0)].reshape(-1, 1))

        # Insert new labels for voxels outside BET mask into mask
        mask[np.where(mask == 0)] = labels

        # Ignore anything that is well below the nose and above top of head
        mask[:, :, 0:50] = 0
        mask[:, :, 300:] = 0

        # Clean up mask
        mask[:, :, 50:300] = morphology.binary_fill_holes(mask[:, :, 50:300])
        mask[:, :, 50:300] = rhino_utils.binary_majority3d(mask[:, :, 50:300])
        mask[:, :, 50:300] = morphology.binary_fill_holes(mask[:, :, 50:300])

        for i in range(mask.shape[0]):
            mask[i, :, 50:300] = morphology.binary_fill_holes(mask[i, :, 50:300])
        for i in range(mask.shape[1]):
            mask[:, i, 50:300] = morphology.binary_fill_holes(mask[:, i, 50:300])
        for i in range(50, 300, 1):
            mask[:, :, i] = morphology.binary_fill_holes(mask[:, :, i])

    # Extract outline
    outline = np.zeros(mask.shape)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = mask.astype(np.uint8)

    # Use morph gradient to find the outline of the solid mask
    for i in range(outline.shape[0]):
        outline[i, :, :] += cv2.morphologyEx(mask[i, :, :], cv2.MORPH_GRADIENT, kernel)
    for i in range(outline.shape[1]):
        outline[:, i, :] += cv2.morphologyEx(mask[:, i, :], cv2.MORPH_GRADIENT, kernel)
    for i in range(50, 300, 1):
        outline[:, :, i] += cv2.morphologyEx(mask[:, :, i], cv2.MORPH_GRADIENT, kernel)
    outline /= 3

    outline[np.where(outline > 0.6)] = 1
    outline[np.where(outline <= 0.6)] = 0
    outline = outline.astype(np.uint8)

    # Save as NIFTI
    outline_nii = nib.Nifti1Image(outline, scalp.affine)
    nib.save(outline_nii, op.join(flirt_outskin_bigfov_file + "_plus_nose.nii.gz"))

    # Command: fslcpgeom <src> <dest>
    fsl_wrappers.fslcpgeom(op.join(flirt_outskin_bigfov_file + ".nii.gz"), op.join(flirt_outskin_bigfov_file + "_plus_nose.nii.gz"))

    # Transform outskin plus nose nii mesh from MNI big FOV to MRI space

    # First we need to invert the flirt_mri2mnibigfov_xform_file xform:
    #
    # Command: convert_xfm -omat <flirt_mnibigfov2mri_xform_file> -inverse <flirt_mri2mnibigfov_xform_file>
    flirt_mnibigfov2mri_xform_file = op.join(filenames["basedir"], "flirt_mnibigfov2mri_xform.txt")
    fsl_wrappers.invxfm(flirt_mri2mnibigfov_xform_file, flirt_mnibigfov2mri_xform_file)  # Note, the wrapper reverses the order of arguments

    # Command: flirt -in <dest> -ref <smri_file> -applyxfm -init <flirt_mnibigfov2mri_xform_file> -out <bet_outskin_plus_nose_mesh_file>
    fsl_wrappers.flirt(
        op.join(flirt_outskin_bigfov_file + "_plus_nose.nii.gz"),
        filenames["smri_file"],
        applyxfm=True,
        init=flirt_mnibigfov2mri_xform_file,
        out=filenames["bet_outskin_plus_nose_mesh_file"],
    )

    # ----------------------------------------------
    # 6) Output the transform from sMRI space to MNI

    flirt_mni2mri = np.loadtxt(mni2mri_flirt_xform_file)
    xform_mni2mri = rhino_utils.get_mne_xform_from_flirt_xform(flirt_mni2mri, filenames["std_brain"], filenames["smri_file"])
    mni_mri_t = Transform("mni_tal", "mri", xform_mni2mri)
    write_trans(filenames["mni_mri_t_file"], mni_mri_t, overwrite=True)

    # ----------------------------------------
    # 7) Output surfaces in sMRI(native) space

    # Transform betsurf output mask/mesh output from MNI to sMRI space
    rhino_utils.transform_bet_surfaces(mni2mri_flirt_xform_file, filenames["mni_mri_t_file"], filenames, filenames["smri_file"])

    # -------------------------------------------
    # Write a file to indicate RHINO has been run

    with open(filenames["completed"], "w") as file:
        file.write(f"Completed: {datetime.now()}\n")
        file.write(f"MRI file: {filenames['smri_file']}\n")
        file.write(f"Include nose: {include_nose}\n")

    # --------
    # Clean up

    if cleanup_files:
        rhino_utils.system_call("rm -f {}".format(op.join(filenames["basedir"], "flirt*")))

    log_or_print('rhino.surfaces.surfaces_display("{}", "{}") can be used to check the result'.format(subjects_dir, subject))
    log_or_print("*** OSL-EPHYS RHINO COMPUTE SURFACES COMPLETE ***")

    return False


def surfaces_display(subjects_dir, subject):
    """Display surfaces.

    Displays the surfaces extracted from the sMRI using rhino.compute_surfaces.

    Display is shown in sMRI (native) space.

    Parameters
    ----------
    subjects_dir : string
        Directory to put RHINO subject directories in. Files will be in subjects_dir/subject/surfaces.
    subject : string
        Subject name directory to put RHINO files in. Files will be in subjects_dir/subject/surfaces.

    Notes
    -----
    bet_inskull_mesh_file is actually the brain surface and bet_outskull_mesh_file is the inner skull surface, due to the naming conventions used by BET.
    """

    filenames = get_surfaces_filenames(subjects_dir, subject)

    rhino_utils.system_call("fsleyes {} {} {} {} {} &".format(
            filenames["smri_file"],
            filenames["bet_inskull_mesh_file"],
            filenames["bet_outskin_mesh_file"],
            filenames["bet_outskull_mesh_file"],
            filenames["bet_outskin_plus_nose_mesh_file"],
        )
    )


def plot_surfaces(subjects_dir, subject, include_nose, already_computed=False):
    """Plot a structural MRI and extracted surfaces.

    Parameters
    ----------
    subjects_dir : str
        Directory to put RHINO subject directories in. Files will be in subjects_dir/subject/surfaces.
    subject : str
        Subject name directory to put RHINO files in. Files will be in subjects_dir/subject/surfaces.
    include_nose : bool
        Specifies whether to add the nose to the outer skin (scalp) surface.
    already_computed : bool, optional
        Have the surfaces (and plots) already been computed?

    Returns
    -------
    output_files : list of str
        Paths to image files saved by this function.
    """
    # Get paths to surface files
    filenames = get_surfaces_filenames(subjects_dir, subject)

    # Surfaces to plot
    surfaces = ["inskull", "outskull", "outskin"]
    if include_nose:
        surfaces.append("outskin_plus_nose")

    # Check surfaces exist
    for surface in surfaces:
        file = Path(filenames[f"bet_{surface}_mesh_file"])
        if not file.exists():
            raise ValueError(f"{file} does not exist")

    # Images to save
    output_files = [f"{filenames['basedir']}/{surface}.png" for surface in surfaces]

    # Check if we need to make plots
    if already_computed:
        if all([Path(file).exists() for file in output_files]):
            return output_files

    # Plot the structural MRI
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress warnings from plotting
        display = nil.plotting.plot_anat(filenames["smri_file"])

    # Plot each surface
    for surface, output_file in zip(surfaces, output_files):
        display_copy = deepcopy(display)
        nii_file = filenames[f"bet_{surface}_mesh_file"]
        img = nil.image.load_img(nii_file)
        data = nil.image.get_data(img)
        vmin = np.nanmin(data)
        vmax = np.nanmax(data)
        display_copy.add_overlay(img, vmin=vmin, vmax=vmax)

        log_or_print(f"Saving {output_file}")
        display_copy.savefig(output_file)

    return output_files