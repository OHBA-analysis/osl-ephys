"""
Coregistration
==============

What is coregistration??
************************
 
In MEG/EEG analysis we have a number of coordinate systems including:
 
* MEG (Device) space - defined with respect to  the MEG dewar.
* Polhemus (Head) space - defined with respect to the locations of the fiducial locations (LPA, RPA and Nasion). The fiducial locations in polhemus space are typically acquired prior to the MEG scan, using a polhemus device.
* sMRI (Native) space - defined with respect to the structural MRI scan.
* MNI space - defined with respect to the MNI standard space brain.
 
In order to compute forward models and carry out source reconstruction we need to have all of the necessary things (MEG sensors, dipoles, scalp surface etc.) in a common coordinate space. However, prior to coregistration, this is not the case. For example, we only have the MEG sensors located in MEG (device) space.
 
Coregistration is the process of learning a mapping between each pair of coordinate systems. To do this, we make use of landmarks (these act as a kind of Rosetta stone) whose locations are known in two different coordinate systems. Knowing where these landmarks are in two coordinate systems allows us to learn a mapping between the coordinate systems.
 
For example, the fiducials (LPA, RPA and Nasion) are known in both sMRI (Native) space and Polhemus (Head) space, and provide the information we need to learn a linear (affine) transform between sMRI (Native) space and Polhemus (Head) space.
 
The different landmarks, and the coordinate systems they are known in prior to coregistration, are summarised here:

Fiducial points
-----------------

A critical stage in the coregistration of polhemus (head) space to sMRI (Native) space is matching the fiducial points (LPA, RPA and nasion) in both spaces. It is therefore important to ensure that one gets as accurate fiducial locations as possible, particularly when using the polhemus system at the point of data acquisition.<br>

Headshape Points
----------------

It is nonetheless challenging to perfectly pinpoint the fiducial locations in both polhemus (head) space to sMRI (Native) space. It is for this reason that we also use headshape points to refine the coregistration. This proceeds as follows:

* learn an initial linear (affine) mapping from polhemus (head) space to sMRI (Native) space using just the fiducials (LPA, RPA and nasion)

* refine this linear (affine) mapping from polhemus (head) space to sMRI (Native) space using the headshape points derived from the polheums system and the scalp surface extracted from the structural MRI

Coregistration using RHINO
**************************

In osl-ephys, the standard approach for coregistration is to use RHINO (Registration of Headshapes Including Nose in osl-ephys). The RHINO pipeline is actually not just used for the coregistration, it can also:

* Compute the head and brain surfaces needed for coregistration and forward modelling from the structural MRI (using FSL tools such as BET and Flirt)
* Perform coregistration so that the MEG sensors and head / brain surfaces can be placed into a common coordinate system.
* Compute the forward model (the lead fields) given the relative geometry of the MEG sensors and head / brain surfaces.

RHINO has a number of key elements that help it produce good coregistrations, including:

* Extraction of the full head surface including the nose. This is a key aspect of RHINO working well. It requires that we can locate the nose in both the structural MRI and in the Polhemus spaces. As such, we make the following recommendations when using RHINO:
    - Ensure a good quality structural MRI that allows extraction of the brain and scalp, and that the nose is included. The sMRI needs a field of view large enough to fully cover the head (NOTE: MRI operators not familiar with MEG often do not realise that the scalp and skull are important for MEG analysis and set the field of view to give maximum resolution of the brain by cropping the skull and scalp!).
    - Acquire a large number (>200) of Headshape Polhemus points. 
    - Acquire Headshape Polhemus points cover the scalp, brow and (rigid parts of) the nose. The surface matching algorithm searches for the best fit of the polhemus points to the surface extracted from the sMRI. Since the scalp is approximately spherical, an apparently good fit of the Polhemus points to the data can be achieved even if the fit is severely rotated away from the true position. This can be avoided by including the nose, which both constrains the fit and makes it easier to determine whether the coregistration has gone awry.
* Performing multi-start optimisation to avoid local minima. The surface matching algorithm is prone to local minima so it can often get stuck with a poor fit. This problem is greater if there are less points from which to estimate the fit. Any misleading/incorrect polhemus points should be removed.
* Extraction of the scalp surface directly from the structural MRI image using FSL tools to avoid distortion (this necessarily assumes the structural is distortionless).
    - Make sure the orientation information in the structural MRI niftii file is correct. Sometimes, the orientation information in the structural MRI nifti file is incorrect, for instance if it has been poorly converted from another format (e.g. DICOM). This can throw the RHINO fitting off, so review the anatomical markers (e.g. anterior/posterior) in FSLeyes and make sure that the sform is set correctly. A good sanity check is to open the structure in FSLeyes, alongside a standard space MNI brain, and make sure that the two images are roughly in the same location and have the same orientation (note - they do not need to be perfectly aligned, the surface extraction call below will do that registration for you).<br>


Running RHINO
=============

Here, we will demonstrate how to run RHINO on some example data from a CTF MEG scanner.

The steps we will follow in this tutorial are:

1. Downloading the data from OSF
2. Setting up the session's files
3. Extracting the surfaces
4. Coregistration
5. Computing the forward model
6. Batched RHINO (surface extraction, coregistration and forward modelling over multiple subjects)

To run this tutorial you will need to have osl-ephys and FSL installed, with the appropriate paths specified in your environment. See the instructions on the repo/read the docs for how to install these packages.

1. Downloading the raw data from OSF
************************************

Let's download the two subjects' data from the OSF project website.

:note: To download the dataset you need ``osfclient`` installed: ``pip install osfclient``.
"""

import os


def get_data(name):
    """Download a dataset from OSF."""
    print('Data will be in directory {}'.format(os.getcwd()))
    if os.path.exists(f"{name}"):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p zxb6c fetch SourceRecon/data/{name}.zip")
    os.system(f"unzip -o {name}.zip")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {name}"

# Download the dataset
get_data("notts_2subjects")

#%%
# 2. Setting up the session's files
# *********************************
# 
# Let's find the structural MRIs and preprocessed fif files on your computer. A :class:`~osl_ephys.session.Session` holds the files RHINO reads and writes for one session: its preprocessed data, the directory of the surfaces extracted from its subject's MRI (``outdir/surfaces/<subject>``) and the coregistration, forward model and beamformer directories under ``outdir/<id>``.

import os.path as op
from pprint import pprint

import numpy as np

from osl_ephys import io, rhino
from osl_ephys.session import Session

data_dir = './notts_2subjects'
outdir = './notts_2subjects/recon'

subject = '{subject}'
smri_files = io.Study(op.join(data_dir, subject, subject + '_T1w.nii.gz')).get()
fif_files = io.Study(op.join(data_dir, subject, subject + '_task-resteyesopen_meg_preproc_raw.fif'))
subjects = fif_files.fields['subject']
fif_files = fif_files.get()

print('Structural files:')
pprint(smri_files)
print('fif files:')
pprint(fif_files)
print('subjects:')
pprint(subjects)

session = Session(
    outdir=outdir,
    id=subjects[0],
    surfaces_dir=op.join(outdir, "surfaces", subjects[0]),
    preproc_file=fif_files[0],
)
print(session)

#%%
# :note: Please change the following directory to wherever you installed fsl (not needed when ``FSLDIR`` is set in your environment)

fsl_dir = '~/fsl'
rhino.setup_fsl(fsl_dir)

#%%
# 3. Extracting the surfaces
# **************************
# 
# The first thing we need to do is to use the structural image to compute the head and brain surfaces in native and MNI space using :func:`~osl_ephys.rhino.extract_surfaces`. The head and brain surfaces will be used later for carrying out the coregistration and forward modelling.
# This step is run once per subject (the sessions of a subject share the surfaces), and can be batched over subjects as we will see later. For now, we will do this for just the first subject. Note that this can take a few minutes.
# 
# The inputs we need to provide are:
# 
# * *mri_file* - the full path to the structural MRI nifti file
# * *outdir* - the surfaces directory of the subject
# * *include_nose* - a boolean flag indicating whether or not to extract a head surface from the structural MRI that includes the nose. If your structural MRI includes the nose AND you have acquired polhemus headshape points that include the nose, then it is recommended to set this flag to True

rhino.extract_surfaces(smri_files[0], session.surfaces_dir, include_nose=True)

#%%
# The surfaces are drawn on slices of the MRI in ``surfaces.png``, next to a plot of the registration to the MNI template (``mni_registration.png``), and the quality of that registration is saved in ``mni_registration.json``. We can also view the result in *fsleyes*. Note that *fsleyes* can sometimes take a few moments to open.
# 
# CHECK that:
# 
# * The surfaces have been extracted properly compared with the structural
# * The nose is included in the scalp surface, if that was requested with the *include_nose* option above
# 
# If there are problems, then check that you have a sufficiently high quality MRI and that the MRI file has the correct orientation information. See more on this in the *Coregistration using RHINO* section above.

print(session.surfaces.surfaces_plot_file)
rhino.fsleyes([
    session.surfaces.mri_file,
    session.surfaces.bet_outskin_mesh_file,
    session.surfaces.bet_inskull_mesh_file,
])

#%%
# 4. Coregistration
# *****************
# 
# Polhemus headshape points
# -------------------------
# 
# Before we run the actual coregistration, we need to provide the coordinates for the nasion, LPA, RPA and the headshape points in Polhemus (head) space, in millimetres. Typically, the polhemus coordinates are extracted from the MEG *fif* file with :func:`~osl_ephys.rhino.extract_fiducials_and_headshape_from_fif` (the ``extract_fiducials_and_headshape`` step of a config), or from a Polhemus ``.pos`` file.
# 
# However, in this practical the polhemus files have been provided in these locations:
# 
# * *./notts_2subjects/sub-not001/polhemus/polhemus_nasion.txt*
# * *./notts_2subjects/sub-not001/polhemus/polhemus_rpa.txt*
# * *./notts_2subjects/sub-not001/polhemus/polhemus_lpa.txt*
# * *./notts_2subjects/sub-not001/polhemus/polhemus_headshape.txt*
# 
# These are ASCII text files that contain space separated (3 x num_coordinates) coordinates (e.g. *polhemus_nasion.txt* contains one column of 3 values). RHINO reads them from the session's coregistration directory, at the paths held by ``session.coreg``:
# 
# * ``session.coreg.head_nasion_file``
# * ``session.coreg.head_rpa_file``
# * ``session.coreg.head_lpa_file``
# * ``session.coreg.head_headshape_file``
# 
# To handle this, we will now define and run a function, *copy_polhemus_files*, that will put the polhemus files of a session at these paths. This function takes the ``dataset`` of a session so that it can also be used as a step of the batch later (where ``Session.from_dataset`` gives the session's files).

def copy_polhemus_files(dataset, userargs):
    session = Session.from_dataset(dataset)
    os.makedirs(session.coreg_dir, exist_ok=True)
    polhemus_dir = op.join(data_dir, session.id, 'polhemus')
    for name in ['nasion', 'rpa', 'lpa', 'headshape']:
        points = np.loadtxt(op.join(polhemus_dir, f'polhemus_{name}.txt'))
        np.savetxt(getattr(session.coreg, f'head_{name}_file'), points)
    return dataset

copy_polhemus_files({"session_files": session}, {})

print('E.g., the coordinates for the nasion for subject {} in Polhemus space are \n'.format(subjects[0]))
print(np.loadtxt(session.coreg.head_nasion_file))

#%%
# We can now perform coregistration so that the MEG sensors and head / brain surfaces can be placed into a common coordinate system.
# 
# We do this by running :func:`~osl_ephys.rhino.coregister_head_and_mri` and passing in:
#
# * *session* - the session's files
# * *use_headshape* - a boolean flag indicating whether or not to use the headshape points to refine the coregistration.
# * *use_nose* - a boolean flag indicating whether or not to use the nose headshape points to refine the coregistration. Setting this to True requires that include_nose was set True in the call to *extract_surfaces*, and requires that the polhemus headshape points include the nose.
# * *n_init* - the number of initialisations of the surface matching; a higher number helps a coregistration that got stuck in a local minimum.

rhino.coregister_head_and_mri(session, use_headshape=True, use_nose=True)

#%%
# The coregistration is drawn in ``coreg.png`` in the coregistration directory, and the distance of each headshape point from the scalp is saved in ``coreg.json`` (``coreg_rms`` gives the root mean square, the number the QC report shows). We can also draw it in an interactive window; note that here we set ``include_nose=False``, which means the nose is not shown in the visualisation even though it is used in the coregistration, because creating a mesh with the nose included is computationally intensive.
# 
# The coregistration result is shown in MEG (device) space (in mm).
# 
# * Grey disks - MEG sensors
# * Blue arrows - MEG sensor orientations
# * Yellow diamonds - MRI-derived fiducial locations
# * Pink spheres - Polhemus-derived fiducial locations
# * Green surface - Whole head scalp extraction
# * Red spheres - Polhemus-derived headshape points
# 
# A good coregistration shows:
# 
# * MRI fiducials (yellow diamonds) in appropriate positions on the scalp
# * Polhemus-derived fiducial locations (pink spheres) in appropriate positions on the scalp 
# * Good correspondence between the headshape points (red spheres) and the scalp
# * The scalp appropriately inside the sensors, and with a sensible orientation.
# 
# If you have a bad co-registration:
# 
# * Go back and check that the surface extraction has worked well using ``fsleyes`` (see above).
# * Check for misleading or erroneous headshape points (red spheres) and remove them. See the `Deleting Headshape Points <https://osl-ephys.readthedocs.io/en/latest/tutorials_build/source-recon_deleting-headshape-points.html>`_ tutorial for how to delete headshape points.
# * Rerun the coregistration with a higher ``n_init``.
# * Check that the settings for using the nose are compatible with the available MRI and headshape points
# * The subject in question may need to be omitted from the ensuing analysis.

print("Coregistration error (mm, rms over the headshape points):", rhino.coreg_rms(session))
rhino.plot_coregistration(session, include_nose=False, show=True)

#%%
# 5. Computing the forward model
# ******************************
# 
# We can now compute the forward model (the lead fields) given we can now put the MEG sensors and head / brain surfaces in the same coordinate system. We do this by running :func:`~osl_ephys.rhino.forward_model`. Note that this is mostly just a wrapper call for a standard MNE function.
# Here we are modelling the brain/head using 'Single Layer', which corresponds to just modelling the inner skull surface, which is the standard thing to do in MEG forward modelling.
# Lead fields will be computed for a regularly spaced dipole grid in MNI space, with a spacing given by the passed in argument *gridstep*. The dipole grid is confined to be inside the brain mask computed by *extract_surfaces*, and the BEM surfaces are checked (and repaired when they intersect) before the model is computed.

rhino.forward_model(session, model="Single Layer", gridstep=10)

#%%
# We now have a *forward model file* stored in the session's coregistration directory.
#     
# This file contains the leadfields that map from source to sensor space, and which are used to do source reconstruction.

from mne import read_forward_solution

fwd = read_forward_solution(session.fwd_model_file)
leadfield = fwd["sol"]["data"]
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

#%%
# 6. Batched RHINO
# ****************
# 
# So far we have shown how we can compute surfaces, do coregistration and forward modelling step-by-step on a single subject at a time.
# Alternatively, we can do batching over multiple subjects. This is much easier to organise in a script and automatically produces the QC report.
# To do this, we need to pass the following to :func:`~osl_ephys.pipeline.run_batch`:
# 
# * *config* - the settings for each of the steps: *extract_surfaces, coregister, forward_model* (in the *coreg* and *source_recon* sections)
# * *sessions* - a dict per session with its *id*, *subject*, *preproc_file* and *smri_file*
# * *outdir* - the output directory where the surfaces, coregistration and forward model are placed
# * *extra_funcs* - this is where we pass the function we wrote earlier *copy_polhemus_files* which will be run to put the polhemus files for each session in the correct locations for RHINO

from osl_ephys.pipeline import run_batch

config = """
    coreg:
        - extract_surfaces:
            include_nose: true
        - copy_polhemus_files: {}
        - coregister:
            use_nose: true
            use_headshape: true
    source_recon:
        - forward_model:
            model: Single Layer
            gridstep: 10
    """

sessions = [
    {"id": subject, "subject": subject, "preproc_file": fif, "smri_file": smri}
    for subject, fif, smri in zip(subjects, fif_files, smri_files)
]
run_batch(config, sessions, outdir, extra_funcs=[copy_polhemus_files])

#%%
# Viewing the batched results
# ---------------------------
# 
# The QC report of the run has a row per session with the coregistration error, and tabs with the surfaces, the MNI registration and the coregistration of the selected session:

print('View the report by opening the following file in a web browser:\n{}'.format(os.getcwd() + ('/notts_2subjects/recon/qc/report.html')))
