"""Surface extraction: the scalp, skull and brain surfaces from a structural MRI with FSL, and their registration to MNI space."""

from __future__ import annotations

import os
import warnings
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.mixture import GaussianMixture
from skimage import measure
from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr, CPointer
from scipy import ndimage, LowLevelCallable
from fsl import wrappers as fsl_wrappers
import mne
from mne.transforms import Transform, write_trans
from mne.surface import _points_outside_surface
from osl_ephys.session import Session, SurfaceFilenames, save_params
from osl_ephys.utils.misc import system_call
from osl_ephys.rhino._geometry import (
    _get_flirt_xform_between_axes,
    _get_mne_xform_from_flirt_xform,
    _get_orient,
    _get_vtk_mesh_native,
    _transform_vtk_mesh,
)


def align_smri_to_mni(
    mri_file: str,
    output_file: str,
    xform_file: str | None = None,
) -> None:
    """Rigidly align a structural MRI to the MNI template.

    For structural MRIs that RHINO cannot use as they are, most commonly a
    image with an "aligned anat" sform code (2), which :py:func:`extract_surfaces
    <osl_ephys.source_recon.rhino.extract_surfaces>` rejects. Registering to the MNI
    template gives a well defined header (sform code 4) to use in its place.

    The registration is rigid, so the head keeps its true size and shape.

    Parameters
    ----------
    mri_file : str
        Full path to the structural MRI to align, in NIfTI format.
    output_file : str
        Full path to write the aligned structural MRI to.
    xform_file : str, optional
        Full path to write the MRI to MNI transform to. Default is None,
        which means the transform is not saved.
    """
    std_brain = f"{os.environ['FSLDIR']}/data/standard/MNI152_T1_1mm.nii.gz"
    kwargs = {} if xform_file is None else {"omat": xform_file}
    fsl_wrappers.flirt(
        mri_file,
        std_brain,
        out=output_file,
        cost="corratio",
        dof=6,
        interp="trilinear",
        **kwargs,
    )


def extract_surfaces(
    mri_file: str,
    outdir: str,
    include_nose: bool = True,
    do_mri2mniaxes_xform: bool = True,
    bet_fval: float | None = None,
    nonlinear_registration: bool = True,
    show: bool = False,
) -> None:
    """Extract surfaces.

    Extracts inner skull, outer skin (scalp) and brain surfaces from passed
    in mri_file, which is assumed to be a T1, using FSL. Assumes that the
    MRI file has a valid sform.

    In more detail:

    1) Transform MRI to be aligned with the MNI axes so that BET works well
    2) Use bet to skull strip MRI so that flirt works well
    3) Use flirt to register skull stripped MRI to MNI space
    4) Use BET/BETSURF to get:

       a) The scalp surface (excluding nose), this gives the MRI-derived
          headshape points in native MRI space, which can be used in the
          headshape points registration later.
       b) The scalp surface (outer skin), inner skull and brain surface, these
          can be used for forward modelling later. Note that due to the unusual
          naming conventions used by BET:

          - bet_inskull_mesh_file is actually the brain surface
          - bet_outskull_mesh_file is actually the inner skull surface
          - bet_outskin_mesh_file is the outer skin/scalp surface

    5) Add nose to scalp surface (optional)
    6) Output the transform from MRI space to MNI
    7) Output surfaces in MRI space
    8) Use FNIRT to nonlinearly register the MRI to MNI space (optional)

    Parameters
    ----------
    mri_file : str
        Full path to structural MRI in NIfTI format (with .nii.gz, .nii,
        .hdr or .mgz extension).
        The sform code must be 1 (Scanner Anat) or 4 (MNI), and the sform
        matrix should transform from voxel indices to coordinates in mm. The
        sform defines the native/MRI coordinate system used throughout RHINO.
        The qform is ignored. You can check the sform code with
        ``fslorient -getsformcode <mri_file>`` and set it with
        ``fslorient -setsformcode 1 <mri_file>``.
    outdir : str
        Output directory.
    include_nose : bool, optional
        Specifies whether to add the nose to the outer skin (scalp) surface.
        This can help RHINO's coregistration to work better, assuming that there
        are headshape points that also include the nose.
        Requires the structural MRI to have a FOV that includes the nose!
    do_mri2mniaxes_xform : bool, optional
        Specifies whether to do step (1) above, i.e. transform MRI to be
        aligned with the MNI axes. Sometimes needed when the MRI goes out
        of the MNI FOV after step (1).
    bet_fval : float, optional
        Fractional intensity threshold for FSL's BET. Default is None, which
        uses BET's default (0.5). Higher values (e.g. 0.6-0.7) give more
        aggressive skull stripping, which can help when the inner skull
        surface includes non-brain tissue.
    nonlinear_registration : bool, optional
        Should we nonlinearly register the MRI to MNI space with FSL's FNIRT
        (initialised with the FLIRT affine registration)? The registration
        is used to transform MNI coordinates into the subject's MRI space,
        e.g. to place the dipoles of the MNI grid in rhino.forward_model and
        in source_recon.virtual_electrodes. A nonlinear registration follows
        the subject's anatomy more closely, so the parcels are placed more
        accurately, but takes a few more minutes. Pass False to use the
        affine registration only (e.g. if the nonlinear registration looks
        wrong for a subject).

        Either way, the registration is checked: the overlap (Dice) of the
        skull-stripped brain with the MNI152 brain mask and the mutual
        information of the MRI with the MNI152 template are saved to
        mni_registration.json and shown on mni_registration.png, which
        plots the edges of the MNI152 template (red) on the registered MRI.
        A warning is printed if the registration looks poor. FNIRT increases
        the mutual information when it works, so if it does not the affine
        registration is used instead.
    show : bool, optional
        Whether to display the surface plots interactively. Default is
        False (suitable for batch processing).
    """

    # Note the jargon used varies for xforms and coord spaces:
    # - MEG (device): dev_head_t --> HEAD (polhemus)
    # - HEAD (polhemus): head_mri_t (polhemus2native) --> MRI (native)
    # - MRI (native): mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices
    # - MRI (native): sform (mri2mniaxes) --> MNI axes

    # RHINO does everything in mm

    print()
    print("Extracting surfaces")
    print("-------------------")

    os.makedirs(outdir, exist_ok=True)
    surfaces = SurfaceFilenames(outdir)

    # Check mri_file
    mri_ext = "".join(Path(mri_file).suffixes)
    if mri_ext not in [".nii", ".nii.gz", ".hdr", ".mgz"]:
        raise ValueError(
            "mri_file needs to have a .nii, .nii.gz, .hdr or .mgz extension"
        )

    # Copy MRI to new file for modification
    img = nib.load(mri_file)
    nib.save(img, surfaces.mri_file)

    # We will always use the sform, and so we will set the qform to be same
    # to stop the original qform from being used by mistake (e.g. by flirt)
    #
    # Command: fslorient -copysform2qform <mri_file>
    fsl_wrappers.misc.fslorient(surfaces.mri_file, copysform2qform=True)

    # Check orientation of the MRI
    mri_orient = _get_orient(surfaces.mri_file)

    if mri_orient != "RADIOLOGICAL" and mri_orient != "NEUROLOGICAL":
        raise ValueError(
            "Cannot determine orientation of brain, please check output of:\n "
            f"fslorient -getorient {surfaces.mri_file}"
        )

    # If orientation is not RADIOLOGICAL then force it to be RADIOLOGICAL
    if mri_orient != "RADIOLOGICAL":
        print("Reorienting brain to be RADIOLOGICAL")

        # Command: fslorient -forceradiological <mri_file>
        fsl_wrappers.misc.fslorient(surfaces.mri_file, forceradiological=True)

    print(
        "You can use the following command line call to check the MRI is "
        "appropriate, including checking that the L-R, S-I, A-P labels are "
        "sensible:"
    )
    print(f"fsleyes {surfaces.mri_file} {surfaces.std_brain_file}")

    # ------------------------------------------------------------------------
    # 1) Transform MRI to be aligned with the MNI axes so that BET works well
    # ------------------------------------------------------------------------

    img = nib.load(surfaces.mri_file)
    img_density = np.sum(img.get_fdata()) / np.prod(img.get_fdata().shape)

    # We will start by transforming MRI so that its voxel indices axes are
    # aligned to MNI's. This helps BET work.

    # Calculate mri2mniaxes
    if do_mri2mniaxes_xform:
        flirt_mri2mniaxes_xform = _get_flirt_xform_between_axes(
            surfaces.mri_file, surfaces.std_brain_file
        )
    else:
        flirt_mri2mniaxes_xform = np.eye(4)

    flirt_mri2mniaxes_xform_file = f"{surfaces.root}/flirt_mri2mniaxes_xform.txt"
    flirt_mri_mniaxes_file = f"{surfaces.root}/flirt_mri_mniaxes.nii.gz"

    def _apply_mri2mniaxes_xform():
        # Write xform to disk so flirt can use it
        np.savetxt(flirt_mri2mniaxes_xform_file, flirt_mri2mniaxes_xform)

        # Apply mri2mniaxes xform to mri to get mri_mniaxes, which means MRIs
        # voxel indices axes are aligned to be the same as MNI's
        # Command: flirt -in <mri_file> -ref <std_brain> -applyxfm \
        #          -init <mri2mniaxes_xform_file> -out <mri_mni_axes_file>
        fsl_wrappers.flirt(
            surfaces.mri_file,
            surfaces.std_brain_file,
            applyxfm=True,
            init=flirt_mri2mniaxes_xform_file,
            out=flirt_mri_mniaxes_file,
        )
        img = nib.load(flirt_mri_mniaxes_file)
        return np.sum(img.get_fdata()) / np.prod(img.get_fdata().shape)

    img_latest_density = _apply_mri2mniaxes_xform()

    if 5 * img_latest_density < img_density:
        # Most of the head is outside the MNI FOV. The sform is valid but its
        # origin is a long way from the head, e.g. the subject was lying low
        # in the scanner. Move the head into the FOV and try again.
        print("Head is outside the MNI FOV, centring it...")
        flirt_mri2mniaxes_xform = _centre_head_in_fov(
            surfaces.mri_file, surfaces.std_brain_file, flirt_mri2mniaxes_xform
        )
        img_latest_density = _apply_mri2mniaxes_xform()

    if 5 * img_latest_density < img_density:
        raise Exception(
            "Something is wrong with the passed in structural MRI: "
            f"{surfaces.mri_file}\n"
            "Either it is empty or the sformcode is incorrectly set.\n\n"
            "Try running the following from a command line:\n"
            f"fsleyes {surfaces.std_brain_file} {surfaces.mri_file}\n\n"
            "And see if the standard space brain is shown in the same postcode "
            "as the structural.\n"
            "If it is not, then the sformcode in the structural image needs "
            "setting correctly.\n"
            "Or try passing do_mri2mniaxes_xform=False."
        )

    # ------------------------------------------------------
    # 2) Use BET to skull strip MRI so that FLIRT works well
    # ------------------------------------------------------

    # Check MRI doesn't contain nans
    # (this can cause segmentation faults with FSL's bet)
    if _check_nii_for_nan(surfaces.mri_file):
        print("WARNING: nan found in MRI file.")

    print("Running BET pre-FLIRT...")

    # Command: bet <flirt_mri_mniaxes_file> <flirt_mri_mniaxes_bet_file>
    flirt_mri_mniaxes_bet_file = f"{surfaces.root}/flirt_mri_mniaxes_bet"
    bet_kwargs = {}
    if bet_fval is not None:
        bet_kwargs["fracintensity"] = bet_fval
    fsl_wrappers.bet(flirt_mri_mniaxes_file, flirt_mri_mniaxes_bet_file, **bet_kwargs)

    # ---------------------------------------------------------
    # 3) Use FLIRT to register skull stripped MRI to MNI space
    # ---------------------------------------------------------

    print("Running FLIRT...")

    # Flirt is run on the skull stripped brains to register the mri_mniaxes
    # to the MNI standard brain
    #
    # Command: flirt -in <flirt_mri_mniaxes_bet_file> -ref <std_brain> \
    #          -omat <flirt_mniaxes2mni_file> -o <flirt_mri_mni_bet_file>
    flirt_mniaxes2mni_file = f"{surfaces.root}/flirt_mniaxes2mni.txt"
    flirt_mri_mni_bet_file = f"{surfaces.root}/flirt_mri_mni_bet.nii.gz"
    fsl_wrappers.flirt(
        flirt_mri_mniaxes_bet_file,
        surfaces.std_brain_file,
        omat=flirt_mniaxes2mni_file,
        o=flirt_mri_mni_bet_file,
    )

    # Calculate overall flirt transform, from MRI to MNI
    #
    # Command: convert_xfm -omat <flirt_mri2mni_xform_file> \
    #          -concat <flirt_mniaxes2mni_file> <flirt_mri2mniaxes_xform_file>
    flirt_mri2mni_xform_file = f"{surfaces.root}/flirt_mri2mni_xform.txt"
    fsl_wrappers.concatxfm(
        flirt_mri2mniaxes_xform_file,
        flirt_mniaxes2mni_file,
        flirt_mri2mni_xform_file,
    )  # Note, the wrapper reverses the order of arguments

    # and also calculate its inverse, from MNI to MRI
    #
    # Command: convert_xfm -omat <mni2mri_flirt_xform_file> \
    #          -inverse <flirt_mri2mni_xform_file>
    mni2mri_flirt_xform_file = surfaces.mni2mri_flirt_xform_file
    fsl_wrappers.invxfm(
        flirt_mri2mni_xform_file, mni2mri_flirt_xform_file
    )  # Note, the wrapper reverses the order of arguments

    # Move full MRI into MNI space to do full bet and betsurf
    #
    # Command: flirt -in <mri_file> -ref <std_brain> -applyxfm \
    #          -init <flirt_mri2mni_xform_file> -out <flirt_mri_mni_file>
    flirt_mri_mni_file = f"{surfaces.root}/flirt_mri_mni.nii.gz"
    fsl_wrappers.flirt(
        surfaces.mri_file,
        surfaces.std_brain_file,
        applyxfm=True,
        init=flirt_mri2mni_xform_file,
        out=flirt_mri_mni_file,
    )

    # ------------------------------------------------------------------------
    # 4) Use BET/BETSURF to get:
    # a) The scalp surface (excluding nose), this gives the MRI-derived
    #    headshape points in native MRI space, which can be used in the
    #    headshape points registration later.
    # b) The scalp surface (outer skin), inner skull and brain surface, these
    #    can be used for forward modelling later. Note that due to the unusual
    #    naming conventions used by BET:
    #    - bet_inskull_mesh_file is actually the brain surface
    #    - bet_outskull_mesh_file is actually the inner skull surface
    #    - bet_outskin_mesh_file is the outer skin/scalp surface
    # ------------------------------------------------------------------------

    print("Running BET and BETSURF...")

    # Run BET and BETSURF on mri to get the surface mesh (in MNI space)
    #
    # Command: bet <flirt_mri_mni_file> <flirt_mri_mni_bet_file> -A
    flirt_mri_mni_bet_file = f"{surfaces.root}/flirt"
    fsl_wrappers.bet(flirt_mri_mni_file, flirt_mri_mni_bet_file, A=True, **bet_kwargs)

    # ---------------------------------------
    # 5) Add nose to scalp surface (optional)
    # ---------------------------------------

    if include_nose:
        print("Refining scalp surface...")

        # We do this in MNI big FOV space, to allow the full nose to be included

        # Calculate flirt_mni2mnibigfov_xform
        mni2mnibigfov_xform = _get_flirt_xform_between_axes(
            from_nii=flirt_mri_mni_file, target_nii=surfaces.std_brain_bigfov_file
        )
        flirt_mni2mnibigfov_xform_file = (
            f"{surfaces.root}/flirt_mni2mnibigfov_xform.txt"
        )
        np.savetxt(flirt_mni2mnibigfov_xform_file, mni2mnibigfov_xform)

        # Calculate overall transform, from mri to MNI big fov
        #
        # Command: convert_xfm -omat <flirt_mri2mnibigfov_xform_file> \
        #          -concat <flirt_mni2mnibigfov_xform_file> <flirt_mri2mni_xform_file>"
        flirt_mri2mnibigfov_xform_file = (
            f"{surfaces.root}/flirt_mri2mnibigfov_xform.txt"
        )
        fsl_wrappers.concatxfm(
            flirt_mri2mni_xform_file,
            flirt_mni2mnibigfov_xform_file,
            flirt_mri2mnibigfov_xform_file,
        )  # Note, the wrapper reverses the order of arguments

        # Move MRI to MNI big FOV space and load in
        #
        # Command: flirt -in <mri_file> -ref <std_brain_bigfov> -applyxfm \
        #          -init <flirt_mri2mnibigfov_xform_file> \
        #          -out <flirt_mri_mni_bigfov_file>
        flirt_mri_mni_bigfov_file = f"{surfaces.root}/flirt_mri_mni_bigfov"
        fsl_wrappers.flirt(
            surfaces.mri_file,
            surfaces.std_brain_bigfov_file,
            applyxfm=True,
            init=flirt_mri2mnibigfov_xform_file,
            out=flirt_mri_mni_bigfov_file,
        )

        # Move scalp to MNI big FOV space and load in
        #
        # Command: flirt -in <flirt_outskin_file> -ref <std_brain_bigfov> \
        #          -applyxfm -init <flirt_mni2mnibigfov_xform_file> \
        #          -out <flirt_outskin_bigfov_file>
        flirt_outskin_file = f"{surfaces.root}/flirt_outskin_mesh"
        flirt_outskin_bigfov_file = f"{surfaces.root}/flirt_outskin_mesh_bigfov"
        fsl_wrappers.flirt(
            flirt_outskin_file,
            surfaces.std_brain_bigfov_file,
            applyxfm=True,
            init=flirt_mni2mnibigfov_xform_file,
            out=flirt_outskin_bigfov_file,
        )
        scalp = nib.load(f"{flirt_outskin_bigfov_file}.nii.gz")

        # Create mask by filling outline

        # Add a border of ones to the mask, in case the complete head is
        # not in the FOV, without this binary_fill_holes will not work
        mask = np.ones(np.add(scalp.shape, 2))

        # Note that z=100 is where the standard MNI FOV starts in the big FOV
        mask[1:-1, 1:-1, 102:-1] = scalp.get_fdata()[:, :, 101:]
        mask[:, :, :101] = 0

        # We assume that the top of the head is not cutoff by the FOV,
        # we need to assume this so that binary_fill_holes works:
        mask[:, :, -1] = 0
        mask = ndimage.morphology.binary_fill_holes(mask)

        # Remove added border
        mask[:, :, :102] = 0
        mask = mask[1:-1, 1:-1, 1:-1]

        print("Adding nose to scalp surface...")

        # Reclassify bright voxels outside of mask (to put nose inside
        # the mask since bet will have excluded it)
        vol = nib.load(f"{flirt_mri_mni_bigfov_file}.nii.gz")
        vol_data = vol.get_fdata()

        # Normalise vol data
        vol_data = vol_data / np.max(vol_data.flatten())

        # Estimate observation model params of 2 class GMM with diagonal
        # cov matrix where the two classes correspond to inside and outside
        # the bet mask
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
        mask[:, :, 50:300] = ndimage.morphology.binary_fill_holes(mask[:, :, 50:300])
        mask[:, :, 50:300] = _binary_majority3d(mask[:, :, 50:300])
        mask[:, :, 50:300] = ndimage.morphology.binary_fill_holes(mask[:, :, 50:300])

        for i in range(mask.shape[0]):
            mask[i, :, 50:300] = ndimage.morphology.binary_fill_holes(
                mask[i, :, 50:300]
            )
        for i in range(mask.shape[1]):
            mask[:, i, 50:300] = ndimage.morphology.binary_fill_holes(
                mask[:, i, 50:300]
            )
        for i in range(50, 300, 1):
            mask[:, :, i] = ndimage.morphology.binary_fill_holes(mask[:, :, i])

        # Extract outline
        outline = np.zeros(mask.shape)
        mask = mask.astype(np.uint8)

        # Use morph gradient to find the outline of the solid mask
        structure = np.ones((3, 3), dtype=bool)

        for i in range(outline.shape[0]):
            slice_bool = mask[i, :, :].astype(bool)
            grad = ndimage.binary_dilation(
                slice_bool, structure=structure
            ) ^ ndimage.binary_erosion(slice_bool, structure=structure)
            outline[i, :, :] += grad.astype(np.uint8)

        for i in range(outline.shape[1]):
            slice_bool = mask[:, i, :].astype(bool)
            grad = ndimage.binary_dilation(
                slice_bool, structure=structure
            ) ^ ndimage.binary_erosion(slice_bool, structure=structure)
            outline[:, i, :] += grad.astype(np.uint8)

        for i in range(50, 300, 1):
            slice_bool = mask[:, :, i].astype(bool)
            grad = ndimage.binary_dilation(
                slice_bool, structure=structure
            ) ^ ndimage.binary_erosion(slice_bool, structure=structure)
            outline[:, :, i] += grad.astype(np.uint8)

        outline /= 3

        outline[np.where(outline > 0.6)] = 1
        outline[np.where(outline <= 0.6)] = 0
        outline = outline.astype(np.uint8)

        # Save as NIFTI
        outline_nii = nib.Nifti1Image(outline, scalp.affine)
        nib.save(outline_nii, f"{flirt_outskin_bigfov_file}_plus_nose.nii.gz")

        # Command: fslcpgeom <src> <dest>
        fsl_wrappers.fslcpgeom(
            f"{flirt_outskin_bigfov_file}.nii.gz",
            f"{flirt_outskin_bigfov_file}_plus_nose.nii.gz",
        )

        # Transform outskin plus nose nii mesh from MNI big FOV to MRI space

        # First we need to invert the flirt_mri2mnibigfov_xform_file xform:
        #
        # Command: convert_xfm -omat <flirt_mnibigfov2mri_xform_file> \
        #          -inverse <flirt_mri2mnibigfov_xform_file>
        flirt_mnibigfov2mri_xform_file = (
            f"{surfaces.root}/flirt_mnibigfov2mri_xform.txt"
        )
        fsl_wrappers.invxfm(
            flirt_mri2mnibigfov_xform_file,
            flirt_mnibigfov2mri_xform_file,
        )  # Note, the wrapper reverses the order of arguments

        # Command: flirt -in <dest> -ref <smri_file> -applyxfm \
        #          -init <flirt_mnibigfov2mri_xform_file> \
        #          -out <bet_outskin_plus_nose_mesh_file>
        fsl_wrappers.flirt(
            f"{flirt_outskin_bigfov_file}_plus_nose.nii.gz",
            surfaces.mri_file,
            applyxfm=True,
            init=flirt_mnibigfov2mri_xform_file,
            out=surfaces.bet_outskin_plus_nose_mesh_file,
        )

    # ----------------------------------------------
    # 6) Output the transform from MRI space to MNI
    # ----------------------------------------------

    flirt_mni2mri = np.loadtxt(mni2mri_flirt_xform_file)
    xform_mni2mri = _get_mne_xform_from_flirt_xform(
        flirt_mni2mri, surfaces.std_brain_file, surfaces.mri_file
    )
    mni_mri_t = Transform("mni_tal", "mri", xform_mni2mri)
    write_trans(surfaces.mni_mri_t_file, mni_mri_t, overwrite=True)

    # ----------------------------------------
    # 7) Output surfaces in MRI (native) space
    # ----------------------------------------

    # Transform betsurf output mask/mesh output from MNI to MRI space
    for mesh_name in ["outskin_mesh", "inskull_mesh", "outskull_mesh"]:
        # xform mask
        #
        # Command: flirt -in <flirt_mesh_file> -ref <mri_file> \
        #          -interp nearestneighbour -applyxfm \
        #          -init <mni2mri_flirt_xform_file> -out <out_file>
        fsl_wrappers.flirt(
            f"{surfaces.root}/flirt_{mesh_name}.nii.gz",
            surfaces.mri_file,
            interp="nearestneighbour",
            applyxfm=True,
            init=mni2mri_flirt_xform_file,
            out=f"{surfaces.root}/{mesh_name}",
        )

        # xform vtk mesh
        _transform_vtk_mesh(
            f"{surfaces.root}/flirt_{mesh_name}.vtk",
            f"{surfaces.root}/flirt_{mesh_name}.nii.gz",
            f"{surfaces.root}/{mesh_name}.vtk",
            f"{surfaces.root}/{mesh_name}.nii.gz",
            surfaces.mni_mri_t_file,
        )

    # ---------------------------------------------------------------------
    # 8) Nonlinearly register the MRI to MNI space with FNIRT (optional) and
    #    check the registration
    # ---------------------------------------------------------------------

    # The registration is checked on the 2 mm MNI grid (the grid FNIRT uses)
    #
    # Command: flirt -in <mri_file> -ref <std_head_2mm> -applyxfm \
    #          -init <flirt_mri2mni_xform_file> -out <mri_mni_affine_file>
    fsl_wrappers.flirt(
        surfaces.mri_file,
        surfaces.std_head_2mm_file,
        applyxfm=True,
        init=flirt_mri2mni_xform_file,
        out=surfaces.mri_mni_affine_file,
    )
    quality = {
        "affine": _mni_registration_quality(
            surfaces, surfaces.mri_mni_affine_file, bet_kwargs
        )
    }

    used = "affine"
    if nonlinear_registration:
        _nonlinear_registration(surfaces)
        quality["nonlinear"] = _mni_registration_quality(
            surfaces, surfaces.mri_mni_nonlinear_file, bet_kwargs
        )
        # FNIRT increases the mutual information when it works, if it does
        # not it has failed and the affine registration is used
        affine_mi = quality["affine"]["mutual_information"]
        if quality["nonlinear"]["mutual_information"] > affine_mi:
            used = "nonlinear"

    if used == "affine":
        # Make sure a warp from FNIRT or a previous call is not used
        for f in [surfaces.mri2mni_warp_file, surfaces.mri_mni_nonlinear_file]:
            if os.path.exists(f):
                os.remove(f)

    _report_mni_registration(surfaces, quality, used)

    print("Cleaning up FLIRT files")
    system_call(f"rm -f {surfaces.root}/flirt*", verbose=False)

    # Plot the surfaces
    plot_surfaces(outdir, include_nose=include_nose)
    if not show:
        plt.close("all")

    print("Surface extraction complete.")


def plot_surfaces(outdir: str, include_nose: bool = True) -> None:
    """Plot the extracted surfaces on the structural MRI.

    The surfaces are drawn on one plot, saved to surfaces.png: the brain
    surface in yellow, the inner skull in cyan and the scalp in magenta. Each
    should follow its anatomical boundary and lie inside the next.

    Parameters
    ----------
    outdir : str
        Directory with the surfaces, see :func:`extract_surfaces`.
    include_nose : bool, optional
        Should we plot the scalp surface including the nose?
    """
    surfaces = SurfaceFilenames(outdir)

    # Surfaces to plot. Note the names BET gives the files: inskull is the
    # brain surface and outskull is the inner skull
    scalp = "outskin_plus_nose" if include_nose else "outskin"
    colours = {
        "Brain": ("inskull", "#FFD400"),
        "Inner skull": ("outskull", "#00C8FF"),
        "Scalp": (scalp, "#FF4FD8"),
    }

    # Check surfaces exist
    for name, _ in colours.values():
        file = Path(getattr(surfaces, f"bet_{name}_mesh_file"))
        if not file.exists():
            raise ValueError(f"{file} does not exist")

    # Plot the structural MRI and each surface, on slices through the middle
    # of the brain (the middle of the MRI can be well below it)
    from nilearn import plotting

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        display = plotting.plot_anat(
            surfaces.mri_file,
            cut_coords=plotting.find_xyz_cut_coords(surfaces.bet_inskull_mesh_file),
            draw_cross=False,
            colorbar=False,
        )
        for name, colour in colours.values():
            display.add_overlay(
                getattr(surfaces, f"bet_{name}_mesh_file"),
                cmap=ListedColormap([colour]),
                threshold=0.5,
            )
    display.frame_axes.legend(
        handles=[
            Line2D([], [], color=colour, label=label)
            for label, (_, colour) in colours.items()
        ],
        loc="lower center",
        ncol=len(colours),
        frameon=False,
        labelcolor="white",
    )

    print(f"Saving {surfaces.surfaces_plot_file}")
    display.savefig(surfaces.surfaces_plot_file, dpi=150)


def repair_bem_surfaces(session: Session, max_iter: int = 50) -> None:
    """Pull stray vertices of the BEM surfaces back inside the surface outside them.

    A boundary element model needs each surface to lie strictly inside the one
    outside it: brain inside inner skull, inner skull inside scalp. BET
    occasionally leaves a handful of vertices spiking through the surface
    outside them, and :func:`mne.make_bem_model` then refuses to build the
    model at all. This only matters for the multi-layer models, i.e. EEG - a
    single layer model uses one surface, so nothing can intersect.

    Each offending vertex is replaced by the mean of its neighbours in the
    mesh, repeated until the surfaces nest. The vertices that are already
    inside are left untouched, so the repair is local to the spikes: it is
    fixing a defect in the segmentation, not reshaping the head.

    The surface files are overwritten, so call this after
    :func:`coregister_head_and_mri` (which writes the copies in the coreg
    directory) and before :func:`forward_model`.

    Parameters
    ----------
    session : Session
        Files of the session.
    max_iter : int, optional
        Maximum number of smoothing passes per surface pair. A spike normally
        clears in a few passes; needing many suggests the segmentation is
        wrong rather than spiky.

    Raises
    ------
    RuntimeError
        If a surface still intersects after ``max_iter`` passes.
    """
    print()
    print("Repairing BEM surfaces")
    print("----------------------")

    coreg = session.coreg

    # Outside to inside. Note RHINO's names are offset from what they hold:
    # bet_outskin is the scalp, bet_outskull the inner skull and bet_inskull
    # the brain surface.
    pairs = [
        (coreg.bet_outskin_surf_file, coreg.bet_outskull_surf_file),
        (coreg.bet_outskull_surf_file, coreg.bet_inskull_surf_file),
    ]

    for outer_file, inner_file in pairs:
        outer_verts, outer_tris = mne.surface.read_surface(outer_file)
        verts, tris = mne.surface.read_surface(inner_file)

        # The test mne.make_bem_model applies: a point is inside a closed
        # surface if the solid angle it subtends is 4*pi. Scale invariant, so
        # it doesn't matter that these surfaces are in mm rather than metres.
        outer_surf = {"rr": outer_verts, "tris": outer_tris}

        def _outside(points):
            return _points_outside_surface(points, outer_surf)

        stray = _outside(verts)
        if not stray.any():
            continue

        # Neighbours of each vertex in the mesh
        neighbours = [[] for _ in range(len(verts))]
        for a, b, c in tris:
            neighbours[a] += [b, c]
            neighbours[b] += [a, c]
            neighbours[c] += [a, b]
        neighbours = [np.unique(n) for n in neighbours]

        original = verts.copy()
        n_stray = int(stray.sum())
        for _ in range(max_iter):
            (indices,) = np.nonzero(stray)
            if not len(indices):
                break
            for index in indices:
                verts[index] = verts[neighbours[index]].mean(axis=0)
            stray = _outside(verts)

        if stray.any():
            raise RuntimeError(
                f"{stray.sum()} vertices of {inner_file} are still outside "
                f"{outer_file} after {max_iter} passes. The segmentation is "
                f"likely to be wrong, check the surfaces."
            )

        moved = np.linalg.norm(verts - original, axis=1).max()
        print(
            f"Overwriting: {inner_file} "
            f"(repaired {n_stray} of {len(verts)} vertices, "
            f"moved by up to {moved:.2f} mm)"
        )
        mne.write_surface(inner_file, verts, tris, overwrite=True)


def _nonlinear_registration(surfaces: SurfaceFilenames) -> None:
    """Nonlinearly register the MRI to MNI space with FNIRT.

    The warp (coefficient file) is saved to surfaces.mri2mni_warp_file and the MRI
    in MNI space to surfaces.mri_mni_nonlinear_file (for checking the
    registration).

    Parameters
    ----------
    surfaces : SurfaceFilenames
        Surface extraction file paths.
    """
    print("Running FNIRT...")

    # FLIRT affine from MRI to MNI to initialise FNIRT with
    mri2mni_flirt_xform = np.linalg.inv(np.loadtxt(surfaces.mni2mri_flirt_xform_file))

    # Command: fnirt --in=<mri_file> --aff=<mri2mni_flirt_xform> \
    #          --ref=<MNI152_T1_2mm> --config=T1_2_MNI152_2mm \
    #          --cout=<mri2mni_warp_file> --iout=<mri_mni_nonlinear_file>
    fsl_wrappers.fnirt(
        surfaces.mri_file,
        aff=mri2mni_flirt_xform,
        ref=surfaces.std_head_2mm_file,
        config="T1_2_MNI152_2mm",
        cout=surfaces.mri2mni_warp_file,
        iout=surfaces.mri_mni_nonlinear_file,
    )


MNI_REGISTRATION_MIN_DICE = 0.85


def _mni_registration_quality(
    surfaces: SurfaceFilenames, registered_file: str, bet_kwargs: dict
) -> dict[str, float]:
    """How well an MRI registered to MNI space matches the MNI152 template.

    Parameters
    ----------
    surfaces : SurfaceFilenames
        Surface extraction file paths.
    registered_file : str
        MRI registered to MNI space, on the 2 mm MNI grid.
    bet_kwargs : dict
        Keyword arguments for BET, as used to extract the surfaces.

    Returns
    -------
    quality : dict
        'dice': overlap of the skull-stripped registered MRI with the MNI152
        brain mask. 'mutual_information': mutual information (in nats) of
        the registered MRI with the MNI152 template within the template
        brain, which increases as the anatomy lines up (unlike a
        correlation, it is not thrown by a bias field).
    """
    # Command: bet <registered_file> <flirt_registered_bet_file> -m
    flirt_registered_bet_file = f"{surfaces.root}/flirt_registered_bet"
    fsl_wrappers.bet(registered_file, flirt_registered_bet_file, m=True, **bet_kwargs)

    def load(file, dtype=float):
        return np.asanyarray(nib.load(file).dataobj, dtype=dtype)

    mask = load(surfaces.std_brain_mask_2mm_file, bool)
    brain = load(f"{flirt_registered_bet_file}_mask.nii.gz", bool)
    dice = 2 * np.sum(brain & mask) / (np.sum(brain) + np.sum(mask))

    # Intensities are scaled to [0, 1] so the histogram bins are the same
    # for every image compared
    def scaled(file):
        x = load(file)[mask]
        return np.clip(x / np.percentile(x, 99.9), 0, 1)

    counts, _, _ = np.histogram2d(
        scaled(registered_file),
        scaled(surfaces.std_head_2mm_file),
        bins=64,
        range=[[0, 1], [0, 1]],
    )
    p = counts / counts.sum()
    independent = p.sum(axis=1, keepdims=True) @ p.sum(axis=0, keepdims=True)
    nonzero = p > 0
    mutual_information = np.sum(p[nonzero] * np.log(p[nonzero] / independent[nonzero]))

    return {"dice": float(dice), "mutual_information": float(mutual_information)}


def _report_mni_registration(
    surfaces: SurfaceFilenames,
    quality: dict[str, dict[str, float]],
    used: str,
) -> None:
    """Save, print and plot the quality of the registration to MNI space.

    Parameters
    ----------
    surfaces : SurfaceFilenames
        Surface extraction file paths.
    quality : dict
        Quality of the 'affine' and (if done) 'nonlinear' registration, from
        _mni_registration_quality.
    used : str
        The registration that will be used, 'affine' or 'nonlinear'.
    """
    save_params(surfaces.mni_registration_quality_file, {"used": used, **quality})

    affine = quality["affine"]
    nonlinear = quality.get("nonlinear")

    print("MNI registration:")
    for name, q in quality.items():
        print(
            f"  {name}: brain overlap (Dice) = {q['dice']:.3f}, mutual "
            f"information with template = {q['mutual_information']:.3f}"
        )
    if quality[used]["dice"] < MNI_REGISTRATION_MIN_DICE:
        print(
            "WARNING: the registration to MNI space looks poor (brain overlap "
            f"{quality[used]['dice']:.2f}). Check "
            f"{surfaces.mni_registration_plot_file}. A cropped field of view, a "
            "strong bias field, a wrong sform or poor skull stripping (bet_fval) "
            "can all cause this."
        )
    if nonlinear and used == "affine":
        print(
            "WARNING: the nonlinear (FNIRT) registration to MNI space is not "
            "better than the affine (FLIRT) one, so the affine registration "
            "is used."
        )

    def summary(q):
        return f"overlap {q['dice']:.2f}, MI {q['mutual_information']:.2f}"

    # The registration that is used comes first
    title = f"FLIRT: {summary(affine)}"
    if used == "nonlinear":
        title = f"FNIRT: {summary(nonlinear)} ({title})"
    elif nonlinear:
        title = f"{title} (FNIRT: {summary(nonlinear)})"
    registered_file = (
        surfaces.mri_mni_nonlinear_file
        if used == "nonlinear"
        else surfaces.mri_mni_affine_file
    )
    _plot_mni_registration(
        registered_file,
        surfaces.std_head_2mm_file,
        surfaces.mni_registration_plot_file,
        title,
    )
    print("You can check the registration with:")
    print(f"fsleyes {surfaces.std_head_2mm_file} {registered_file}")


def _plot_mni_registration(
    registered_file: str, mni_file: str, output_file: str, title: str
) -> None:
    """Plot the edges of the MNI template on the MRI registered to MNI space.

    Parameters
    ----------
    registered_file : str
        MRI registered to MNI space.
    mni_file : str
        MNI template the MRI was registered to.
    output_file : str
        Output filename.
    title : str
        Title for the plot.
    """
    from nilearn import plotting

    data = np.asanyarray(nib.load(registered_file).dataobj)
    vmax = np.percentile(data[data > 0], 99)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        display = plotting.plot_anat(
            registered_file,
            display_mode="ortho",
            cut_coords=(0, -18, 18),
            title=title,
            draw_cross=False,
            dim=0,
            vmin=0,
            vmax=vmax,
            colorbar=False,
        )
        display.add_edges(mni_file, color="r")
    print(f"Saving {output_file}")
    display.savefig(output_file)
    display.close()


def _create_freesurfer_meshes_from_bet_surfaces(
    session: Session, xform_mri_voxel2mri: np.ndarray
) -> None:
    """
    Create sMRI-derived freesurfer surfaces in native/mri space in mm,
    for use by forward modelling
    """
    _create_freesurfer_mesh_from_bet_surface(
        infile=session.bet_inskull_mesh_vtk_file,
        surf_outfile=session.bet_inskull_surf_file,
        nii_mesh_file=session.bet_inskull_mesh_file,
        xform_mri_voxel2mri=xform_mri_voxel2mri,
    )
    _create_freesurfer_mesh_from_bet_surface(
        infile=session.bet_outskull_mesh_vtk_file,
        surf_outfile=session.bet_outskull_surf_file,
        nii_mesh_file=session.bet_outskull_mesh_file,
        xform_mri_voxel2mri=xform_mri_voxel2mri,
    )
    _create_freesurfer_mesh_from_bet_surface(
        infile=session.bet_outskin_mesh_vtk_file,
        surf_outfile=session.bet_outskin_surf_file,
        nii_mesh_file=session.bet_outskin_mesh_file,
        xform_mri_voxel2mri=xform_mri_voxel2mri,
    )


def _create_freesurfer_mesh_from_bet_surface(
    infile: str,
    surf_outfile: str,
    xform_mri_voxel2mri: np.ndarray,
    nii_mesh_file: str | None = None,
) -> None:
    """Creates surface mesh in .surf format and in native mri space in mm from infile.

    Parameters
    ----------
    infile : str
        Either:
        1) .nii.gz file containing zero's for background and one's for surface
        2) .vtk file generated by bet_surf (in which case the path to the
        structural MRI, smri_file, must be included as an input)
    surf_outfile : str
        Path to the .surf file generated, containing the surface mesh in mm
    xform_mri_voxel2mri : np.ndarray
        4x4 array. Transform from voxel indices to native/mri mm.
    nii_mesh_file : str, optional
        Path to the niftii mesh file that is the niftii equivalent of vtk file
        passed in as infile (only needed if infile is a vtk file).
    """
    pth, name = os.path.split(infile)
    name, ext = os.path.splitext(name)

    if ext == ".gz":
        print("Creating surface mesh for {} .....".format(infile))

        name, ext = os.path.splitext(name)
        if ext != ".nii":
            raise ValueError("Invalid infile. Needs to be a .nii.gz or .vtk file")

        # Load NIfTI and binarize
        nii = nib.load(infile)
        vol = nii.get_fdata()
        # Ensure binary mask (0 background, >0 surface)
        vol = (vol > 0).astype(np.uint8)

        # Run marching cubes
        # level=0.5 extracts the surface between 0 and 1
        # spacing left as (1,1,1) because we will apply a full 4x4 transform below.
        try:
            verts_vox, faces, normals, values = measure.marching_cubes(
                vol, level=0.5, spacing=(1.0, 1.0, 1.0)
            )
        except Exception as e:
            raise RuntimeError(
                "marching_cubes failed. Check that the NIfTI file is a "
                "proper binary mask."
            ) from e

        if verts_vox.size == 0 or faces.size == 0:
            raise RuntimeError(
                "marching_cubes produced no vertices/faces. Check input volume/mask."
            )

        # verts_vox is (M,3) in voxel coordinates (voxel index space)
        # Convert to homogeneous coordinates and apply the provided 4x4 transform
        ones = np.ones((verts_vox.shape[0], 1), dtype=verts_vox.dtype)
        verts_vox_h = np.hstack([verts_vox, ones])  # shape (M,4)

        # Ensure xform is numpy array and has shape (4,4)
        xform = np.asarray(xform_mri_voxel2mri)
        if xform.shape != (4, 4):
            raise ValueError("xform_mri_voxel2mri must be a 4x4 array")

        verts_mm_h = (xform @ verts_vox_h.T).T  # (M,4)
        verts_mm = verts_mm_h[:, :3]

        # faces already has shape (F,3) and is integer
        faces = faces.astype(int)

        # Write FreeSurfer surface
        mne.surface.write_surface(
            surf_outfile, verts_mm, faces, file_format="freesurfer", overwrite=True
        )

    elif ext == ".vtk":
        if nii_mesh_file is None:
            raise ValueError(
                "You must specify a nii_mesh_file (niftii format) "
                "if infile format is vtk"
            )

        rrs_native, tris_native = _get_vtk_mesh_native(infile, nii_mesh_file)

        mne.surface.write_surface(
            surf_outfile,
            rrs_native,
            tris_native,
            file_format="freesurfer",
            overwrite=True,
        )

    else:
        raise ValueError("Invalid infile. Needs to be a .nii.gz or .vtk file")


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def _majority(values_ptr, len_values, result, data):
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


def _binary_majority3d(img: np.ndarray) -> np.ndarray:
    """
    Set a pixel to 1 if a required majority (default=14) or more pixels
    in its 3x3x3 neighborhood are 1, otherwise, set the pixel to 0. img
    is a 3D binary image
    """
    if img.dtype != "bool":
        raise ValueError("binary_majority3d(img) requires img to be binary")
    if len(img.shape) != 3:
        raise ValueError("binary_majority3d(img) requires img to be 3D")
    return ndimage.generic_filter(
        img, LowLevelCallable(_majority.ctypes), size=3
    ).astype(int)


def _centre_head_in_fov(
    from_nii: str, target_nii: str, flirt_xform: np.ndarray
) -> np.ndarray:
    """
    Translates a flirt xform so that the head in from_nii lands inside the FOV
    of target_nii.

    The head is centred left-right and front-back and its top is put just below
    the top of the FOV, which is where the head is in the MNI template.

    flirt_xform must put from_nii on the same axes as target_nii, with the last
    axis running from the bottom to the top of the head (see
    _get_flirt_xform_between_axes), and both images must be RADIOLOGICAL.
    """
    step = 4  # mm resolution is plenty for finding the head
    img = nib.load(from_nii)
    data = np.asarray(img.dataobj)[::step, ::step, ::step]

    # Voxels in the head, with the intensity threshold BET uses
    lower, upper = np.percentile(data, [2, 98])
    voxels = np.argwhere(data > lower + 0.1 * (upper - lower)) * step

    # Where they land in target_nii, in flirt coords (mm)
    coords = voxels * np.array(img.header.get_zooms()[:3])
    coords = coords @ flirt_xform[:3, :3].T + flirt_xform[:3, 3]

    target = nib.load(target_nii)
    fov = (np.array(target.shape[:3]) - 1) * target.header.get_zooms()[:3]

    # The top of the head, robust to noise above it, and the head without the
    # neck and shoulders
    top = np.percentile(coords[:, 2], 99)
    head = coords[coords[:, 2] > top - 120]

    centred_xform = flirt_xform.copy()
    centred_xform[:2, 3] += fov[:2] / 2 - head[:, :2].mean(axis=0)
    centred_xform[2, 3] += fov[2] - 15 - top
    return centred_xform


def _check_nii_for_nan(filename: str) -> bool:
    """Check nii file for nans."""
    img = nib.load(filename)
    data = img.get_fdata()
    return np.isnan(data).any()
