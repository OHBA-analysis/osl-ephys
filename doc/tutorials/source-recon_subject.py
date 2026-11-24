"""
Single Subject Source Reconstruction Tutorial
=============================================

In this tutorial, we will step through how to do source reconstruction (and parcellation) on a single session of data from a subject in the Wakeman-Henson task MEG dataset. This is a public dataset consisting of 19 healthy individuals who performed a simple visual perception task. See `Wakeman & Henson <https://www.nature.com/articles/sdata20151>`_ for more details.
 
The steps we will follow are:
 
1. Downloading the data from OSF
2. Setup file names
3. Compute surfaces, perform coregistration, and compute forward model using batching
4. Temporal Filtering
5. Compute beamformer weights
6. Apply beamformer weights
7. Parcellation
8. Epoching

Note that most of these steps can be carried out more simply over multiple subjects using batching. See the "Group Analysis of Source-space Data" tutorial for an example of this. Here we break the steps down and run them manually, to give you insight into how it all works.

To run this tutorial you will need to have osl-ephys and FSL installed, with the appropriate paths specified in your environment. See the instructions on the repo/read the docs for how to install these packages.


1. Downloading the raw data from OSF
************************************

The public Wakeman-Henson dataset provides MaxFiltered data. Note that the full dataset is available on `OpenNeuro <https://openneuro.org/datasets/ds000117/versions/1.0.4>`_. Here, we will work with just a single subject from this dataset, which can be downloaded from the OSF project website. 

Let's download the data. Note, to download the dataset you need ``osfclient`` installed. This can be installed by executing the following code in a jupyter notebook cell:

``!pip install osfclient``

We can now download the data for the single subject we will look at. Note that this will be placed in your current working directory.

"""


import os

def get_data(name):
    print('Data will be in directory {}'.format(os.getcwd()))
    """Download a dataset from OSF."""
    if os.path.exists(f"{name}"):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p zxb6c fetch SourceRecon/data/{name}.zip")
    os.system(f"unzip -o {name}.zip")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {name}"

# Download the dataset
get_data("wake_hen")

#%%
# 2. Setup File Names
# *******************
# Let's first setup all the file names we will need, for an example single session from a single subject from the Wakeman-Henson dataset. 

import os.path as op
from pprint import pprint

from osl_ephys import io

# setup dirs
data_dir = './wake_hen'
recon_dir = op.join(data_dir, "recon")

# structurals
sub_name = "sub{sub_num}"
smri_files = io.Study(op.join(data_dir, sub_name, "anatomy", "highres001.nii.gz")).get()

# fif files
subject = "{subject}"
preproc_fif_files = io.Study(op.join(data_dir, subject + "_meg", subject + "_meg_preproc_raw.fif"))
subjects = preproc_fif_files.fields['subject']
preproc_fif_files = preproc_fif_files.get()

print('subjects:')
pprint(subjects)
print('Structural files:')
pprint(smri_files)
print('Preproc fif files:')
pprint(preproc_fif_files)

#%%
# 3. Compute Surfaces, Coregistration and Forward Modelling
# *********************************************************
#
# Here, we set the options in the ``config``, and use :func:`~osl_ephys.pipeline.run_batch`.
# See the tutorial on "Coregistration with RHINO" for more on how this works.
# 
# We do not use the nose and headshape points as these were not acquired for this dataset.
# 
# Setting ``gridstep: 10`` means that the data will be source reconstructed to each point on a regular 3D grid, with spacings of 10mm.

from osl_ephys import rhino
from osl_ephys.pipeline import run_batch

config = """
    coreg:
    - extract_surfaces:
        include_nose: false
    - extract_fiducials_and_headshape: {}
    - coregister:
        use_nose: false
        use_headshape: false
    source_recon:
    - forward_model:
        model: Single Layer
        gridstep: 10
"""

fsl_dir = '~/fsl'
rhino.setup_fsl(fsl_dir)

sessions = [
    {"id": subject, "subject": subject, "preproc_file": fif, "smri_file": smri}
    for subject, fif, smri in zip(subjects, preproc_fif_files, smri_files)
]
run_batch(config, sessions, recon_dir)

# The files of the session we reconstruct by hand below
from osl_ephys.session import Session

session = Session(
    outdir=recon_dir,
    id=subjects[0],
    surfaces_dir=op.join(recon_dir, "surfaces", subjects[0]),
    preproc_file=preproc_fif_files[0],
)

#%%
# 4. Temporal Filtering
# *********************
# 
# We temporally filter the data to focus on the oscillatory content that we are interest in. 
# 
# Here, we assume that we will be doing an evoked response (ERF) analysis on the epoched task data, and so we filter to the frequency range where the evoked response is typically contained, i.e. between 1 and 30 Hz.
# 

import mne

chantypes = ["grad"]

# Get and setup the data
data = mne.io.read_raw_fif(preproc_fif_files[0], preload=True)
data = data.pick(chantypes)

# Filter to the beta band
print("Temporal Filtering")
data = data.filter(
    l_freq=3,
    h_freq=20,
    method="iir",
    iir_params={"order": 5, "btype": "bandpass", "ftype": "butter"},
)
print("Completed")

#%%
# 5. Compute beamformer weights
# *****************************
#
# We now compute the beamformer weights (aka filters). These are computed using the (sensors x sensors) data covariance matrix estimated from the preprocessed and the temporally filtered MEG data (contained in *data*), and the forward model of the session. The filters are saved in the session's source directory (``session.filters_file``).
# 
# Note that this automatically ignores any bad time segments when calculating the beamformer filters.
# 
# Here we source reconstructing using just the gradiometers.
# 
# The MEG data in the Wakeman and Henson dataset has been maxfiltered and so the maximum rank is ~64. We therefore slightly conservatively set the rank to be 55. This is used to regularise the estimate of the data covariance matrix.
# 
# More generally, a dipole is a 3D vector in space. Setting ``pick_ori="max-power-pre-weight-norm"`` means that we are computing a scalar beamformer, by projecting this 3D vector on the direction in which there is maximum power. 

from osl_ephys import parcellation
from osl_ephys.source_recon import beamforming

# Make LCMV beamformer filters
# Note that this will exclude any bad time segments when calculating the beamformer filters
beamforming.lcmv_beamformer(
    session,
    data,
    chantypes,
    pick_ori="max-power-pre-weight-norm",
    rank={"grad": 55},
)

#%%
# 6. Applying beamformer weights
# ******************************
#
# We now apply the beamformer filters to the data to project the data into source space, on the dipole grid in MNI space.
# 
# Note that although the beamformer filters were calculated by ignoring any bad time segments, we apply the filters to all time points including the bad time segments. This will make it easier to do epoching later.

print("Applying beamformer spatial filters")

# The source space time series (dipoles x time points) and the MNI coordinates of the dipoles
recon_timeseries_mni, recon_coords_mni = beamforming.apply_lcmv_beamformer(
    session, data, reject_by_annotation=None, reference_brain="mni"
)

print("Completed")
print("Dimensions of reconstructed timeseries in MNI space is (dipoles x all_tpts) = {}".format(recon_timeseries_mni.shape))

#%%
# 7. Parcellation
# ******************
# 
# At this point, the data has been source reconstructed to dipoles (in this case, a scalar value) at each point on a regular 3D grid, with spacings of 10mm. We could then analyse the data across all these dipoles.
# 
# An alternative, is to map the data onto a brain parcellation. This reduces the number of samples in the space from number of dipoles down to number of parcels. Using a parcellation helps to boost the signal to noise ratio, boost correspondence between subjects, reduce the severity of multiple comparison correction when doing any statistics, and aids anatomical interpretability.
# 
# The parcellation we use here is a combination of cortical regions from the Harvard Oxford atlas, and selected sub-cortical regions from the Schaefer 100 parcellation. 
# 
# Let's take a look at the positions of the centres of each parcel in the parcellation.

parcellation_fname = 'atlas-SchaeferHarvOxf_nparc-114_space-MNI_res-8x8x8.nii.gz'

# plot centre of mass for each parcel
p = parcellation.plot_parcellation(parcellation_fname)

#%%
# Compute Parcel Time-courses
# ---------------------------
# 
# We use this parcellation to compute the parcel time courses using the parcellation and the dipole time courses. Note that the output parcel timepoints includes all time points, including any bad time segments.
# 
# Let's now parcellate the data to compute parcel time courses. This is done using the "pca" method, where the parcel time-course is the first principal component over all voxels in the parcel, weighted by the spatial map for the parcel (see `here <https://pubmed.ncbi.nlm.nih.gov/25862259/>`_).
#
# Apply parcellation to (voxels x all_tpts) data contained in recon_timeseries_mni.
# The resulting parcel_timeseries will be (parcels x all_tpts) in MNI space
# where all_tpts includes bad time segments

parcel_ts = parcellation.parcellate(
    recon_timeseries_mni,
    recon_coords_mni,
    parcellation_fname,
    method="pca",
)

#%%
# We now put the parcel time courses into a new MNE raw object *parc_raw*. This will allow us to easily perform epoching using MNE.

# reload raw data to ensure that the stim channel is in there
raw = mne.io.read_raw_fif(preproc_fif_files[0])

parc_raw = parcellation.convert_to_mne_raw(parcel_ts, raw, extra_chans="stim")
print("Dimensions of parc_raw are (nparcels x all_tpts) = {}".format(parc_raw.get_data().shape))

#%%
# 8. Epoching
# ***********
# 
# We can now perform epoching. Note that any epochs (aka trials) that contain any bad time segments will be rejected at this point.

from osl_ephys import preproc

dataset = preproc.read_dataset(preproc_fif_files[0])
epochs = mne.Epochs(
    parc_raw,
    dataset["events"],
    dataset["event_id"],
    tmin=-1,
    tmax=3,
    baseline=(None, 0),
)

print("Dimensions of epochs are (good_epochs x parcels x tpts_within_epoch) = {}".format(epochs.get_data().shape))

#%%
# We can now plot a simple evoked response for this session of data, by averaging over all epochs (aka trials), for a selected parcel

import matplotlib.pyplot as plt
import numpy as np

parcel_ind = 5
print("Plotting group COPE time course for parcel:", parcel_ind)

# average over trials/epochs
erf = np.mean(epochs.get_data()[:, parcel_ind, :], axis=0)

plt.figure()
plt.plot(epochs.times, erf)
plt.title("ERF, for parcel={}".format(parcel_ind))
plt.xlabel("time (s)")
plt.ylabel("ERF")
