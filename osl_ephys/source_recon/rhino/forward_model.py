"""Forward modelling in RHINO.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import os.path as op
from copy import deepcopy

from mne import make_bem_model, make_bem_solution, make_forward_solution, write_forward_solution
from mne.bem import ConductorModel, read_bem_solution
from mne.transforms import read_trans, Transform
from mne.io import read_info
from mne.io.constants import FIFF
from mne.surface import read_surface, write_surface

try:
    from mne.source_space import _make_volume_source_space, _complete_vol_src
except ImportError:
    # Depreciated in mne 1.6
    from mne.source_space._source_space import _make_volume_source_space, _complete_vol_src

import osl_ephys.source_recon.rhino.utils as rhino_utils
from osl_ephys.source_recon.rhino import get_coreg_filenames
from osl_ephys.utils.logger import log_or_print


def forward_model(
    subjects_dir,
    subject,
    model="Single Layer",
    gridstep=8,
    mindist=4.0,
    exclude=0.0,
    eeg=False,
    meg=True,
    verbose=False,
):
    """Compute forward model.

    Parameters
    ----------
    subjects_dir : string
        Directory to find RHINO subject dirs in.
    subject : string
        Subject name dir to find RHINO files in.
    model : string
        'Single Layer' or 'Triple Layer'
        'Single Layer' to use single layer (brain/cortex)
        'Triple Layer' to three layers (scalp, inner skull, brain/cortex)
    gridstep : int
        A grid will be constructed with the spacing given by ``gridstep`` in mm generating a volume source space.
    mindist : float
        Exclude points closer than this distance (mm) to the bounding surface.
    exclude : float
        Exclude points closer than this distance (mm) from the center of mass of the bounding surface.
    eeg : bool
        Whether to compute forward model for eeg sensors
    meg : bool
        Whether to compute forward model for meg sensors
    """
    log_or_print("*** RUNNING RHINO VOLUMETRIC FORWARD MODEL ***")

    filenames = rhino_utils.get_rhino_filenames(subjects_dir, subject)

    # Compute MNE bem solution
    if model == "Single Layer":
        conductivity = (0.3,)  # for single layer
    elif model == "Triple Layer":
        conductivity = (0.3, 0.006, 0.3)  # for three layers
    else:
        raise ValueError("{} is an invalid model choice".format(model))

    vol_src = setup_volume_source_space(subjects_dir, subject, gridstep=gridstep, mindist=mindist, exclude=exclude)

    # The BEM solution requires a BEM model which describes the geometry of the head the conductivities of the different tissues.
    # See: https://mne.tools/stable/auto_tutorials/forward/30_forward.html#sphx-glr-auto-tutorials-forward-30-forward-py
    #
    # Note that the BEM does not involve any use of transforms between spaces. The BEM only depends on the head geometry and conductivities.
    # It is therefore independent from the MEG data and the head position.
    #
    # This will get the surfaces from: subjects_dir/subject/bem/inner_skull.surf, which is where rhino.setup_volume_source_space will have put it.

    model = make_bem_model(subjects_dir=subjects_dir, subject=subject, ico=None, conductivity=conductivity, verbose=verbose)
    bem = make_bem_solution(model)
    fwd = make_fwd_solution(subjects_dir, subject, src=vol_src, ignore_ref=True, bem=bem, eeg=eeg, meg=meg, verbose=verbose)
    write_forward_solution(filenames["fwd_model"], fwd, overwrite=True)

    log_or_print("*** OSL-EPHYS RHINO FORWARD MODEL COMPLETE ***")


def make_fwd_solution(
    subjects_dir,
    subject,
    src,
    bem,
    meg=True,
    eeg=True,
    mindist=0.0,
    ignore_ref=False,
    n_jobs=1,
    verbose=None,
):
    """Calculate a forward solution for a subject. This is a RHINO wrapper for mne.make_forward_solution.

    See mne.make_forward_solution for the full set of parameters, with the exception of:

    Parameters
    ----------
    subjects_dir : string
        Directory to find RHINO subject dirs in.
    subject : string
        Subject name dir to find RHINO files in.

    Returns
    -------
    fwd : instance of Forward
        The forward solution.

    Notes
    -----
    Forward modelling is done in head space.

    The coords of points to reconstruct to can be found in the output here:

    >>> fwd['src'][0]['rr'][fwd['src'][0]['vertno']]

    where they are in head space in metres.

    The same coords of points to reconstruct to can be found in the input here:

    >>> src[0]['rr'][src[0]['vertno']]

    where they are in native MRI space in metres.
    """
    filenames = get_coreg_filenames(subjects_dir, subject)

    # src should be in MRI space. Let's just check that is the case
    if src[0]["coord_frame"] != FIFF.FIFFV_COORD_MRI:
        raise RuntimeError("src is not in MRI coordinates")

    # --------------------------------------------
    # Setup main MNE call to make_forward_solution

    # The forward model is done in head space
    # We need the transformation from MRI to HEAD coordinates (or vice versa)
    head_scaledmri_trans_file = filenames["head_scaledmri_t_file"]
    if isinstance(head_scaledmri_trans_file, str):
        head_mri_t = read_trans(head_scaledmri_trans_file)
    else:
        head_mri_t = head_scaledmri_trans_file

    # RHINO does everything in mm, so need to convert it to metres which is what MNE expects.
    # To change units on an xform, just need to change the translation part and leave the rotation alone
    head_mri_t["trans"][0:3, -1] = head_mri_t["trans"][0:3, -1] / 1000

    # Get bem solution
    if isinstance(bem, str):
        bem = read_bem_solution(bem)
    else:
        if not isinstance(bem, ConductorModel):
            raise TypeError("bem must be a string or ConductorModel")
        bem = bem.copy()

    for i in range(len(bem["surfs"])):
        bem["surfs"][i]["tris"] = bem["surfs"][i]["tris"].astype(int)

    # Load fif info
    info_fif_file = filenames["info_fif_file"]
    info = read_info(info_fif_file)

    # -------------
    # Main MNE call

    fwd = make_forward_solution(
        info,
        trans=head_mri_t,
        src=src,
        bem=bem,
        eeg=eeg,
        meg=meg,
        mindist=mindist,
        ignore_ref=ignore_ref,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    # fwd should be in Head space. Let's just check that is the case:
    if fwd["src"][0]["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
        raise RuntimeError("fwd['src'][0] is not in HEAD coordinates")

    return fwd


def setup_volume_source_space(subjects_dir, subject, gridstep=5, mindist=5.0, exclude=0.0):
    """Set up a volume source space grid inside the inner skull surface.

    This is a RHINO specific version of mne.setup_volume_source_space.

    Parameters
    ----------
    subjects_dir : string
        Directory to find RHINO subject dirs in.
    subject : string
        Subject name dir to find RHINO files in.
    gridstep : int
        A grid will be constructed with the spacing given by ``gridstep`` in mm generating a volume source space.
    mindist : float
        Exclude points closer than this distance (mm) to the bounding surface.
    exclude : float
        Exclude points closer than this distance (mm) from the center of mass of the bounding surface.

    Returns
    -------
    src : :py:class:`mne.SourceSpaces`
        A single source space object.

    See Also
    --------
    :py:func:`mne.setup_volume_source_space`


    Notes
    -----
    This is a RHINO specific version of mne.setup_volume_source_space, which can handle smri's that are niftii files.
    This specifically uses the inner skull surface in:

    >>> get_coreg_filenames(subjects_dir, subject)['bet_inskull_surf_file']

    to define the source space grid.

    This will also copy the:

    >>> get_coreg_filenames(subjects_dir, subject)['bet_inskull_surf_file']

    file to:

        ``subjects_dir/subject/bem/inner_skull.surf`

    since this is where mne expects to find it when mne.make_bem_model is called.

    The coords of points to reconstruct to can be found in the output here:

    >>> src[0]['rr'][src[0]['vertno']]

    where they are in native MRI space in metres.
    """
    filenames = get_coreg_filenames(subjects_dir, subject)

    # Note that due to the unusal naming conventions used by BET and MNE:
    # - bet_inskull_*_file is actually the brain surface
    # - bet_outskull_*_file is actually the inner skull surface
    # - bet_outskin_*_file is the outer skin/scalp surface
    #
    # These correspond in mne to (in order):
    # - inner_skull
    # - outer_skull
    # - outer_skin
    #
    # This means that for single shell model, i.e. with conductivities set to length one, the surface used by MNE willalways be the inner_skull,
    # i.e. it actually corresponds to the brain/cortex surface!! Not sure that is correct/optimal.
    #
    # Note that this is done in Fieldtrip too!, see the "Realistic single-shell model, using brain surface from segmented mri" section at:
    # https://www.fieldtriptoolbox.org/example/make_leadfields_using_different_headmodels/#realistic-single-shell-model-using-brain-surface-from-segmented-mri
    #
    # However, others are clear that it should really be the actual inner surface of the skull, see the "single-shell Boundary Element Model (BEM)" bit at:
    # https://imaging.mrc-cbu.cam.ac.uk/meg/SpmForwardModels
    #
    # To be continued... ?

    # ---------------------------------------------------------------------------------------------------------------
    # Move the surfaces to where MNE expects to find them for the forward modelling, see make_bem_model in mne/bem.py

    # First make sure bem directory exists
    bem_dir = op.join(subjects_dir, subject, "bem")
    os.makedirs(bem_dir, exist_ok=True)

    # Note that the coreg surf files are in scaled MRI space
    verts, tris = read_surface(filenames["bet_inskull_surf_file"])
    tris = tris.astype(int)
    write_surface(op.join(bem_dir, "inner_skull.surf"), verts, tris, file_format="freesurfer", overwrite=True)
    log_or_print("Using bet_inskull_surf_file for single shell surface")

    verts, tris = read_surface(filenames["bet_outskull_surf_file"])
    tris = tris.astype(int)
    write_surface(op.join(bem_dir, "outer_skull.surf"), verts, tris, file_format="freesurfer", overwrite=True)

    verts, tris = read_surface(filenames["bet_outskin_surf_file"])
    tris = tris.astype(int)
    write_surface(op.join(bem_dir, "outer_skin.surf"), verts, tris, file_format="freesurfer", overwrite=True)

    # ------------------------------------------------
    # Setup main MNE call to _make_volume_source_space

    pos = float(int(gridstep))
    pos /= 1000.0  # convert pos to m from mm for MNE

    vol_info = rhino_utils.get_vol_info_from_nii(filenames["smri_file"])

    surface = op.join(bem_dir, "inner_skull.surf")
    surf = read_surface(surface, return_dict=True)[-1]
    surf = deepcopy(surf)
    surf["rr"] *= 1e-3  # must be in metres for MNE call

    # -------------
    # Main MNE call

    sp = _make_volume_source_space(surf, pos, exclude, mindist, filenames["smri_file"], None, vol_info=vol_info, single_volume=False)
    sp[0]["type"] = "vol"

    # ----------------------
    # Save and return result

    sp = _complete_vol_src(sp, subject)

    # Add dummy mri_ras_t and vox_mri_t transforms as these are needed for the forward model to be saved (for some reason)
    sp[0]["mri_ras_t"] = Transform("mri", "ras")
    sp[0]["vox_mri_t"] = Transform("mri_voxel", "mri")

    if sp[0]["coord_frame"] != FIFF.FIFFV_COORD_MRI:
        raise RuntimeError("source space is not in MRI coordinates")

    return sp
