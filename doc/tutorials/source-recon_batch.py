"""
Group Analysis of Source-space Data
===================================

In this tutorial, we will perform a group analysis of parcellated source-space task MEG data using the Wakeman-Henson dataset. This is a public dataset consisting of 19 healthy individuals who performed a simple visual perception task. See `Wakeman & Henson (2015) <https://www.nature.com/articles/sdata20151>`_ for more details.

We will do this on the raw time courses bandpass filtered between 3-14Hz. In other words, this will tell us about the strength of the visual evoked response band-limited to be from between 3Hz to 14Hz.

The steps are:

1. Downloading the data from OSF
2. Setup file names
3. Coreg, Source reconstruction and parcellation 
4. Epoching
5. First-Level GLM 
6. Group-Level GLM

To run this tutorial you will need to have osl-ephys and FSL installed, with the appropriate paths specified in your environment. See the instructions on the repo/read the docs for how to install these packages. Before running this tutorial, we recommend going through the **Source reconstruction** and **Statistics (General Linear Modelling)** tutorials first.

1. Downloading the raw data from OSF
************************************

The public Wakeman-Henson dataset provides MaxFiltered data. Note that the full dataset is available on `OpenNeuro <https://openneuro.org/datasets/ds000117/versions/1.0.4>`_.

The full dataset contains 19 subjects, each with 6 sessions. To limit the amount of data that we have to handle for the tutorial, we will use only the first 3 sessions from each of the 19 subjects. This data can be downloaded from the OSF project website. 

Let's download the data. Note, to download the dataset you need osfclient installed. This can be installed by executing the following code in a jupyter notebook cell:

``!pip install osfclient``

Let's now download the data. Note that this will be placed in your current working directory.


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
get_data("wake_hen_group_raw")

#%%
# 2. Setup File Names
# *******************
# 
# Let's first setup the file names for the first 3 sessions from each of the subjects. 
# Note that in the original publication, subjects 1,5 and 16 were excluded from analysis. We will do the same here.


import os
import os.path as op
import numpy as np

from osl_ephys import rhino

fsl_dir = '~/fsl'
rhino.setup_fsl(fsl_dir)

subjects_dir = "./wake_hen_group_raw"
out_dir = op.join(subjects_dir, "glm")

nsubjects = 19
nsessions = 3 # we will only use 3 of the 6 session/runs available from each subject
subjects_to_do = np.arange(0, nsubjects)
sessions_to_do = np.arange(0, nsessions)
subj_sess_2exclude = np.zeros([nsubjects, nsessions]).astype(bool)

subj_sess_2exclude[0]=True
subj_sess_2exclude[4]=True
subj_sess_2exclude[15]=True


preproc_fif_files = []
input_fif_files = []
epoch_fif_files = []
glm_model_files = []
glm_time_files = []
subj_indices = []

recon_dir = op.join(subjects_dir, "recon")
glm_dir = op.join(subjects_dir, "glm")

if not os.path.isdir(glm_dir):
    os.makedirs(glm_dir)

for sub in subjects_to_do:
    for ses in sessions_to_do:
        if not subj_sess_2exclude[sub, ses]:

            sub_name = "sub" + ("{}".format(subjects_to_do[sub] + 1)).zfill(3)
            ses_name = "run_" + ("{}".format(sessions_to_do[ses] + 1)).zfill(2)
            subject = sub_name + "_" + ses_name

            # output files
            preproc_fif_file = op.join(
                subjects_dir, subject + "_meg", subject + "_meg_preproc_raw.fif"
            )
            input_fif_file = op.join(
                recon_dir, subject, "sflip_parc-raw.fif"
            )
            epoch_fif_file = op.join(
                recon_dir, subject, "epoch_sflip_parc-epo.fif"
            )
            glm_model_file = op.join(
                glm_dir, subject, "first_level_glm_model.hdf5"
            )
            glm_time_file= op.join(
                glm_dir, subject, "first_level_glm_model_times.npy"
            )

            if op.exists(epoch_fif_file):
                preproc_fif_files.append(preproc_fif_file)
                input_fif_files.append(input_fif_file)
                epoch_fif_files.append(epoch_fif_file)
                glm_model_files.append(glm_model_file)
                glm_time_files.append(glm_time_file)

                # store which subject this session belongs to,
                # this will be used to construct the group design matrix
                subj_indices.append(sub)

                glm_subj_dir = op.join(glm_dir, subject)
                if not os.path.isdir(glm_subj_dir):
                    os.makedirs(glm_subj_dir)

print(epoch_fif_files)

#%%
# 3. Coreg, Source Reconstruction and Parcellation
# ************************************************
# 
# See the "Single Subject Source Reconstruction" tutorial for an explanation of the settings used here.
# 
# Here we are using ``osl_ephys.pipeline.run_batch`` to easily run Coreg, Source Reconstruction and Parcellation over all subjects and sessions. The sessions are dicts with an ``id`` and their ``preproc_file`` and ``smri_file``.
# 
# Note that we do not actually run this code here. For the sake of time, it has already been run for you (it is actually not possible to run this code as the necessary files have not been provided).


pre_run = True

if not pre_run:

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
    - beamform:
        frequency_range: [3, 20]
        chantypes: [mag, grad]
        rank: {meg: 55}
    parcellation:
    - parcellate:
        parcellation_file: atlas-SchaeferHarvOxf_nparc-114_space-MNI_res-8x8x8.nii.gz
    group:
    - sign_flip: {}
    """
    sessions = [
        {"id": subject, "subject": subject[:6], "preproc_file": preproc, "smri_file": smri}
        for subject, preproc, smri in zip(subjects, preproc_fif_files, smri_files)
    ]
    run_batch(config, sessions, recon_dir)
else:
    print('Using pre-run results')

#%%
# View Coregistration Report
# --------------------------
# 
# The QC report of the batch call, with the coregistration of every session, can be viewed in a web browser.


print('View the report by opening the following file in a web browser:\n{}'.format(os.getcwd() + ('/wake_hen_group_raw/recon/qc/report.html')))


#%%
# 4. Epoching
# ***********
#
# We next loop over subjects, epoching the data 
# 
# Note that we do not actually run this code here. For the sake of time, it has already been run for you (it is actually not possible to run this code as the necessary files have not been provided).


if not pre_run:

    for preproc_fif_file, sflip_parc_file, epoch_fif_file \
            in zip(preproc_fif_files, sflip_parc_files, epoch_fif_files):

        # Parcellated data
        raw = mne.io.read_raw(sflip_parc_file) 
        
        # To get epoching info
        dataset = osl_ephys.preproc.read_dataset(preproc_fif_file)
        
        epochs = mne.Epochs(
            raw,
            dataset["events"],
            dataset["event_id"],
            tmin=-0.2,
            tmax=1.3,
            baseline=(None, 0),
        )

        epochs.drop_bad(verbose=True)
        epochs.load_data()
        epochs.save(epoch_fif_file, overwrite=True)
else:
    print('Using pre-run results')



#%%
# 5. First-level GLM
# ******************
#
# Setup First-level Design Matrix
# -------------------------------
#
# Recall that we have 19 subjects, each with 3 sessions of data. In the experiment there are 9 different types of trials corresponding to 9 different conditions (i.e. different types of visual stimuli) that are presented on a video screen to the subject in the scanner. The 9 different conditions are:
# 
# * FamousFirst
# * FamousImmediate
# * FamousLast
# * UnfamiliarFirst
# * UnfamiliarImmediate
# * UnfamiliarLast
# * ScrambledFirst
# * ScrambledImmediate
# * ScrambledLast
# 
# The *First-Level* analysis corresponds to separately modelling what is happening in the data of each session of each subject. We do this using a "Trial-wise" GLM, because the regressors in the design matrix of the GLM explain the variability over trials.
# 
# Note that we will fit a trial-wise GLM **separately** to each:
# 
# * session
# * parcel
# * time point within trial (or epoch)
# 
# We will now specify the content of the first-level design matrix using the package glmtools.
# 
# Note that we specify 9 regressors, each of which is a categorical regressor that picks out those trials that correspond to each of the 9 different conditions.
# 
# We also specify 2 contrasts:
# 
# * ``Faces_vs_Scrambled`` contrast, which computes the difference in the response between the conditions in which a person's face is presented versus those in which a scrambled face is presented. 
# * ``Visual`` contrast, which computes a contrast that sums over all the 9 different conditions; this therefore corresponds to the average response over all conditions



import glmtools as glm

DC = glm.design.DesignConfig()
DC.add_regressor(name="FamousFirst", rtype="Categorical", codes=5)
DC.add_regressor(name="FamousImmediate", rtype="Categorical", codes=6)
DC.add_regressor(name="FamousLast", rtype="Categorical", codes=7)
DC.add_regressor(name="UnfamiliarFirst", rtype="Categorical", codes=13)
DC.add_regressor(name="UnfamiliarImmediate", rtype="Categorical", codes=14)
DC.add_regressor(name="UnfamiliarLast", rtype="Categorical", codes=15)
DC.add_regressor(name="ScrambledFirst", rtype="Categorical", codes=17)
DC.add_regressor(name="ScrambledImmediate", rtype="Categorical", codes=18)
DC.add_regressor(name="ScrambledLast", rtype="Categorical", codes=19)
DC.add_contrast(
    name="Faces_vs_Scrambled",
    values={
        "FamousFirst": 1,
        "FamousImmediate": 1,
        "FamousLast": 1,
        "UnfamiliarFirst": 1,
        "UnfamiliarImmediate": 1,
        "UnfamiliarLast": 1,
        "ScrambledFirst": -2,
        "ScrambledImmediate": -2,
        "ScrambledLast": -2,
    },
)
DC.add_contrast(
    name="Visual",
    values={
        "FamousFirst": 1,
        "FamousImmediate": 1,
        "FamousLast": 1,
        "UnfamiliarFirst": 1,
        "UnfamiliarImmediate": 1,
        "UnfamiliarLast": 1,
        "ScrambledFirst": 1,
        "ScrambledImmediate": 1,
        "ScrambledLast": 1,
    },
)

print(DC.to_yaml())

#%%
# Each session will have a design matrix containing the 9 different regressors and the 2 contrasts on these regressors specified above. However, because the ordering of trials might be different for each session, we need to construct the regressors in a manner that is specific to each session.
# 
# Hence, we construct the design matrix for an example session.


import mne

subj_index = 0

# Load data in glmtools
epochs = mne.read_epochs(epoch_fif_files[subj_index]) 
data = glm.io.load_mne_epochs(epochs)

# Create design matrix for this session
des = DC.design_from_datainfo(data.info)

print('Completed')

#%%
# We can visualise the resulting first-level, trial-wise design matrix for the example session.
# 
# You should be able see that each regressor (column) is a categorical regressor picking out which trials correspond to each of the 9 different conditions. Below the design matrix you should also see the two contrasts that we will compute.



print('First-level design matrix for subject {}'.format(subj_index))
fig = des.plot_summary()

#%%
# Fit First-level GLM
# -------------------
# 
# We next loop over all sessions, constructing and fitting the first-level design matrices to each run (session) from each subject separately.


import h5py

for epoch_fif_file, glm_model_file, glm_time_file \
        in zip(epoch_fif_files, glm_model_files, glm_time_files):
    
    epochs = mne.read_epochs(epoch_fif_file) # e.g. sensor, source space, or parcellated data
    epochs.load_data()

    # Load data in glmtools
    data = glm.io.load_mne_epochs(epochs)
    
    # Create design matrix for this session
    design = DC.design_from_datainfo(data.info)

    # Fit Model
    model = glm.fit.OLSModel(design, data)

    # Save fitted GLM
    out = h5py.File(glm_model_file, "w")
    design.to_hdf5(out.create_group("design"))
    model.to_hdf5(out.create_group("model"))
    out.close()
    np.save(glm_time_file, epochs.times)

print('Completed')

#%%
# 6. Group-level GLM
# ******************
# 
# Group Model
# -----------
#
# We could perform a group analysis by simply computing a group average of the response over all sessions and subjects. However, it is more flexible to use a GLM at the group level too. We do this using a "Session/subject-wise" GLM, because the regressors in the design matrix of the group-level GLM explain the variability over session and subjects.
# 
# Note that typically the same group-level design matrix is fit *separately* to each:
# 
# * first-level contrast
# * parcel
# * timepoint within trial
# 
# We will now setup the group-level design matrix. We basically have one categorical regressor (it contains zeros and ones) for each subject that picks out which sessions belong to that subject. As such, in the visualisation of the design matrix below you will see 19 regressors for the 19 subjects, each indicating with a value of one which sessions belong to that subject.
# 
# We also need to setup contrasts on the regression parameters, to compute the COPEs (contrasts of parameter estimates) we are interested in. 
# 
# Here, we specify a contrast for each subject, which picks out the the regressor for that subject; and we have one contrast that computes a COPE proportional to the average of the response over all subjects (contrast 20 in the visualisation below - although note that this indexes from 1, whereas when we want to select this context in the python code later, it will have an index of 19, because python indexes from 0).




groupDC = glm.design.DesignConfig()

# Add subject mean regressors
zero_values = dict()
ones_values = dict()
for subj_ind in set(subj_indices):
    regressor_name=("Subj{}".format(subj_ind))
    groupDC.add_regressor(name=regressor_name, rtype="Categorical", codes=subj_ind)
    zero_values[regressor_name]=0
    ones_values[regressor_name]=1

# Add subject mean contrasts
for subj_ind in set(subj_indices):
    contrast_name=("Subj{}".format(subj_ind))
    subj_values = zero_values.copy()
    subj_values[contrast_name] = 1
    groupDC.add_contrast(name=contrast_name,
                    values=subj_values,
                   )
# Add group mean contrast
groupDC.add_contrast(name="Group mean",
                values=ones_values,
               )

design = groupDC.design_from_datainfo({'category_list':subj_indices})

fig = design.plot_summary()

#%%
# As mentioned above, we can fit this group-level design matrix *separately* to each:
# 
# * first-level contrast
# * parcel
# * timepoint within trial 
# 
# Here, we will focus on fitting the group-level GLM to just the first-level contrast with index 1. This corresponds to the average over all conditions (and therefore over all trials). 




first_level_contrast = 1 # indexing starts from 0

print(DC.contrast_names[first_level_contrast])
print(DC.contrasts[first_level_contrast])

#%%
# This output shows that the COPE this first-level contrast computes, is proportional to the average over all conditions (and therefore over all trials). In other words, we can use this first-level contrast to get the response averaged over all trials in each session. 
# 
# Load First-level COPES
# ----------------------
#
# We start by loading in the first-level COPEs from the first-level GLM that we fit earlier, and concatenate them into a (sessions x parcels x tpts_within_trial) array.




import matplotlib.pyplot as plt
from anamnesis import obj_from_hdf5file

rectify = True
baseline_correct = True
baseline_window = (-0.2, 0) # secs

print("Loading first-level GLMs and extracting COPEs for first-level contrast {}".format(first_level_contrast))

data = []
for glm_time_file, glm_model_file in zip(glm_time_files, glm_model_files):

    # Load GLM
    model = obj_from_hdf5file(glm_model_file, "model")
    epochs_times = np.load(glm_time_file)

    baseline_time_inds = np.where((epochs_times>baseline_window[0]) & (epochs_times<baseline_window[1]))[0]

    cope = model.copes[first_level_contrast, :, :]

    if rectify:
        cope = abs(cope)

    if baseline_correct:
        baseline_mean = np.mean(
            cope[:, baseline_time_inds],
            axis=1,
        )
        cope = cope - np.reshape(baseline_mean, [-1, 1])

    data.append(cope)

first_level_copes_data = np.asarray(data) # (sessions x parcels x tpts_within_trial)

# Create GLM data
first_level_copes = glm.data.TrialGLMData(data=first_level_copes_data)
print("Complete")

#%%
# **NOTE**: Sign Ambiguity 
# ------------------------
# There is a complication when comparing evoked responses across sessions and parcels (or dipoles, if we were working at the level of dipoles rather than parcels), caused by an ambiguity in the signs of the parcel time courses. In short, due to the way in which we source reconstruct and compute parcel time courses, we can not tell whether or not the values in a particular parcel and session should have their signs flipped. 
# 
# This is a problem, for example, when:
# 
# 1) Pooling effects over sessions/subjects. This is because some sessions/subject may have their parcel time courses flipped one way, and other sessions/subjects their parcel time courses flipped the other way. Without solving this issue, averaging over sessions/subjects would not work.  
# 
# 2) Comparing an effect across parcels. This is because some parcels may have their time courses flipped one way, and other parcels have their time courses flipped the other way. Without solving this issue, spatial maps that show an effect as it changes over parcels might not look sensible.
# 
# A solution we can use to solve this problem, is to rectify (take the absolute value) of the first-level COPE time courses. This has been carried out in the cell above by setting:
# ``rectify = True``
# 
# This means that we will fit the group-level model to the **absolute value** of the first-level COPEs.
#
# Fitting the Group Model
# -----------------------
#
# We can now fit the group-level GLM to the (sessions x parcels x tpts_within_trial) array, *data*, that contains the absolute value of the first-level COPEs. Essentially, the subject-wise, group-level design matrix will be fit separately to the first-level COPEs for every combination of parcels and timepoints-within-trial. 




# Fit Model
print("Fitting group-level GLM for first-level contrast {}".format(first_level_contrast))
group_model = glm.fit.OLSModel(design, first_level_copes)
print("Complete")

#%%
# Output and view COPE 
# --------------------
#
# Here, we will output and view the parcel-wise COPEs for the first group-level contrast (index 0) of the first-level contrast specified above. 
# 
# We first create a 3D niftii object in MNI space at a time point of interest. 
# 



from osl_ephys import rhino
from osl_ephys.files.functions import find_file
from osl_ephys.parcellation import parcellation
from nilearn import plotting

# index for group contrast we want to output
group_contrast_ind = group_model.copes.shape[0]-1 # group mean

# time point of interest:
tpt = 0.14 # in seconds
volume_num = np.abs(epochs_times-tpt).argmin() # finds index of nearest epoch time to tpt

# The parcellation niftii file needs to be the same as was used to do the parcellation, 
# although it does not need to be at the same spatial resolution as the one used there.
parcellation_file = 'atlas-SchaeferHarvOxf_nparc-114_space-MNI_res-2x2x2.nii.gz'
mask_file = "MNI152_T1_2mm_brain.nii.gz"

cope_map = group_model.copes[group_contrast_ind, :, volume_num]

# Create niftii object
nii = parcellation.convert2niftii(cope_map, find_file(parcellation_file), find_file(mask_file))

print('Created 3D Niftii object for group contrast {}'.format(group_contrast_ind))

#%%
# Let's now view the 3D niftii object as a *png* image file, which shows the parcel-wise COPEs on the cortical surface.




# Setup stats dir to put results into
stats_dir = op.join(subjects_dir, "glm_stats")
if not op.isdir(stats_dir):
    os.makedirs(stats_dir, exist_ok=True)

cope_fname = op.join(
    stats_dir,
    "cope_gc{}_fc{}_vol{}".format(group_contrast_ind, first_level_contrast, volume_num),
)

plotting.plot_img_on_surf(
    nii,
    views=["lateral", "medial"],
    hemispheres=["left", "right"],
    colorbar=True,
    output_file=cope_fname,
)

os.system('open {}'.format(cope_fname + '.png'))

print("Complete")


# We can also create a 3D niftii file, which can then be viewed using *fsleyes*.

import nibabel as nib

nib.save(nii, cope_fname + '.nii.gz')
rhino.fsleyes([find_file(mask_file), cope_fname + '.nii.gz'])

#%%
# In ``fsleyes``:
# 
# * Set the positive colormap from ``Greyscale`` to ``Red-Yellow``
# * Turn on the negative colormap, and change it from ``Greyscale`` to ``Blue-LightBlue``
# * Set ``Min`` to 60, and ``Max`` to 150
# 
#
# Plot time course of group COPE 
# ------------------------------
#
# Let's plot the group-averaged evoked response timecourse for a specified parcel in the visual cortex, alongside time courses for all sessions (19 subjects * 3 sessions per subject). The group-averaged evoked response timecourse corresponds to the group contrast with an index of 19, as defined earlier.



parcel_ind = 53 # indexes from 0

first_level_data = first_level_copes.data  # nsess x nparcels x ntpts

# we divide the group mean COPE by nsubjects to get an average, 
# as the COPE was defined earlier as the sum over all subjects
group_mean = group_model.copes[group_contrast_ind, parcel_ind, :].T/nsubjects  

plt.figure()
plt.plot(epochs_times, first_level_data[:, parcel_ind, :].T)
plt.plot(epochs_times, group_mean, linewidth=2, color='k')
plt.axvline(0, linestyle="--", color="black")

plt.title(
    "abs(cope) for first-level contrast {}, parcel={}".format(
        first_level_contrast, parcel_ind
    )
)
plt.xlabel("time (s)")
plt.ylabel("abs(cope)")

#%%
# A time of 0 secs corresponds to when the visual stimulus was presented. Each line in the plot corresponds to one of the ``3runs x 19subjects = 57sessions``, and shows the abs(COPE) timecourse from the chosen parcel for that session. This shows that there is a huge amount of between-session variability in the cope time course, around the group mean over subjects (which is shown as the black line). As a result, the black line does not look very much like a classic evoked response (ERP or ERF)!
# 
# Let's now look to see how much of this variability is caused by between-subject differences by plotting each subject's mean timecourse. Note that each subject's mean timecourse is available as one of the group contrasts. 
# We will only plot a few subjects, to stop the plot becoming too cluttered:




from matplotlib.pyplot import cm

parcel_ind = 53 # indexes from 0, parcel in visual cortex
time_inds = np.where((epochs_times>-0.2) & (epochs_times<1.3))[0]

# for better visualisation we will only plot the subject means for a few subjects
subjects2plot = [11,12,13,14,15]

# get mean for each subject
subject_means = group_model.copes[:, parcel_ind, time_inds].T
subject_means = subject_means[:, subjects2plot]

# compute standard deviation over sessions for each subject
within_subject_stddev = np.sqrt(nsessions*group_model.varcopes[:, parcel_ind, time_inds]).T
within_subject_stddev = within_subject_stddev[:, subjects2plot]

# we divide the group mean COPE by nsubjects to get an average, 
# as the COPE was defined earlier as the sum over all subjects
group_mean = group_model.copes[group_contrast_ind, parcel_ind, time_inds].T/nsubjects  

clrs = cm.rainbow(np.linspace(0, 1, len(subjects2plot)))

plt.figure()

for sub_ind in range(len(subjects2plot)):
    plt.plot(epochs_times[time_inds], 
             subject_means[:, sub_ind],
             c=clrs[sub_ind])
    plt.fill_between(epochs_times[time_inds], 
                 subject_means[:, sub_ind]-within_subject_stddev[:, sub_ind], 
                 subject_means[:, sub_ind]+within_subject_stddev[:, sub_ind],
                 alpha=0.3, 
                 facecolor=clrs[sub_ind])

plt.plot(epochs_times[time_inds], group_mean, linewidth=2, color='k')
plt.axvline(0, linestyle="--", color="black")

plt.title(
    "abs(cope) for contrast {}, parcel={}".format(
        first_level_contrast, parcel_ind
    )
)
plt.xlabel("time (s)")
plt.ylabel("abs(cope)")

#%%
# The group mean is again shown as the black line. Each subject's mean cope time course is shown plus/minus one stddev of the session population.
# 
# This shows that the overall between session variability is dominated by between subject variability, particularly in the timing of the peaks. While this issue is ameliorated by the use of the abs(cope), an alternative that helps further with this problem, is to do an analysis on the amplitude of the time course (e.g. computed using a Hilbert or Wavelet transform). Please see the "Group Analysis on Amplitude Source-space Data" tutorial for an example of this.
# 
# Statistics
# ----------
# 
# When we computed the first-level COPEs we subtracted the average baseline COPE value. This means that these baseline-corrected first-level COPEs are be expected to be zero if the activity is the same as the baseline period. This means that we can do a statistical test on the group mean COPE, for which any significant time points correspond to time points where the activity is different to the baseline period.
# 
# We will do a 2-tailed test, which finds where the group mean COPE is significantly larger, or smaller, than zero. 
# 
# We will do this using permutation statistics on just the visual cortex parcel that we have already been looking at. We will also focus on a smaller time window to reduce unnecessarily excessive multiple comparison correction.




# Let's do permutation stats on a focussed time window to reduce excessive multiple comparison correction
perm_time_inds = np.where((epochs_times>-0.1) & (epochs_times<0.8))[0]
first_level_copes_data_4perms = first_level_copes_data.copy()

first_level_copes_data_4perms = first_level_copes_data_4perms[:, parcel_ind, perm_time_inds]
print("data for stats is (subjects x timepoints) = {}".format(first_level_copes_data_4perms.data.shape))

# we divide the group mean COPE by nsubjects to get an average, 
# as the COPE was defined earlier as the sum over all subjects
first_level_copes_4perms = glm.data.TrialGLMData(data=first_level_copes_data_4perms/nsubjects)
perm = glm.permutations.MaxStatPermutation(design, 
                                           first_level_copes_4perms,         
                                           contrast_idx=group_contrast_ind, # this is the group mean contrast
                                           nperms=1000, 
                                           metric="copes", 
                                           tail=0, # 2-tailed test
                                           pooled_dims=1, # pool null distribution over time
                                          )
thres = perm.get_thresh(95)  # p-value=0.05

print("threshold:", thres)

#%%
# Now we have a threshold for a p-value of 0.05, let's see which time points in the evoked response are significant.
# These timepoints are shown by solid horizental black lines at the bottom of the plot.




parcel_ind = 53 # indexes from 0, parcel in visual cortex
time_inds = np.where((epochs_times>-0.2) & (epochs_times<1.3))[0]

# we divide the group mean COPE by nsubjects to get an average, 
# as the COPE was defined earlier as the sum over all subjects
group_mean = group_model.copes[group_contrast_ind, parcel_ind, time_inds].T/nsubjects

# Plot time points that are significant
significant = (group_mean > thres) | (group_mean < -thres)

# for better visualisation we will only plot a few subjects
subjects2plot = [11,12,13,14,15]

# compute mean over sessions for each subject
subject_means = group_model.copes[:, parcel_ind, time_inds].T
subject_means = subject_means[:, subjects2plot]

# compute standard deviation over sessions for each subject
within_subject_stddev = np.sqrt(nsessions*group_model.varcopes[:, parcel_ind, time_inds]).T
within_subject_stddev = within_subject_stddev[:, subjects2plot]

clrs = cm.rainbow(np.linspace(0, 1, len(subjects2plot)))

plt.figure()

for sub_ind in range(len(subjects2plot)):
    plt.plot(epochs_times[time_inds], 
             subject_means[:, sub_ind],
             c=clrs[sub_ind])
    plt.fill_between(epochs_times[time_inds], 
                 subject_means[:, sub_ind]-within_subject_stddev[:, sub_ind], 
                 subject_means[:, sub_ind]+within_subject_stddev[:, sub_ind],
                 alpha=0.3, 
                 facecolor=clrs[sub_ind])
    
plt.plot(epochs_times[time_inds], group_mean, linewidth=2, color='k')

sig_times = epochs_times[time_inds][significant]

if len(sig_times) > 0:
    y = -5
    plt.plot((sig_times.min(), sig_times.max()), (y, y), color='k', linewidth=4)    

plt.axvline(0, linestyle="--", color="black")
plt.title(
    "abs(cope) for contrast {}, parcel={}".format(
        first_level_contrast, parcel_ind
    )
)
plt.xlabel("time (s)")
plt.ylabel("abs(cope)")

#%%
# View group average as 4D niftii file 
# ------------------------------------
# 
# Earlier, we viewed a 3D volume of the parcel-wise COPEs as a 3D niftii object at a time point of interest. 
# 
# We will now view the parcel-wise COPEs over all timepoints within the trial, by outputting the parcel-wise COPEs as a 4D niftii object, where the 4th dimension is timepoint within trial.
# 



import nibabel as nib

cope_map = group_model.copes[group_contrast_ind, :, time_inds]

nii = parcellation.convert2niftii(cope_map, 
                                  find_file(parcellation_file), 
                                  find_file(mask_file), 
                                  tres=epochs_times[1]-epochs_times[0], 
                                  tmin=epochs_times[time_inds[0]])

cope_fname = op.join(
    stats_dir,
    "cope_gc{}_fc{}".format(group_contrast_ind, first_level_contrast, volume_num),
)

# Save cope as nii file and view in fsleyes
print(f"Saving {cope_fname}")
nib.save(nii, cope_fname + '.nii.gz')

rhino.fsleyes([find_file(parcellation_file), find_file(mask_file), cope_fname + '.nii.gz'])
#%%
# In ``fsleyes``:
# 
# * Change the positive colormap from *Greyscale* to *Red-Yellow*
# * Turn on the negative colormap, and change it from *Greyscale* to *Blue-LightBlue*
# * Set *Min* to 50, and *Max* to 150
# * Set *Volume* index to 52
# * Click on a voxel in the primary visual cortex
# * From the drop down menus Select *View/Time series*
# 
# To see the x-axis of the time series plots in secs, rather than by index:
# 
# * In the Time series panel, select Settings (the spanner icon)
# * In the Time series settings popup, select "Use Pix Dims"
# 
# Fsleyes only shows time via the volume index (i.e. it is not in seconds) in the ortho-view.
# To convert from volume index to time in seconds, or vice versa, use the following:



vol_index = 52 # indexes from 0
print('vol index of {}, corresponds to {} secs'.format(vol_index, epochs_times[time_inds[vol_index]]))

t = 0.6 # secs
vol_index = np.abs(epochs_times[time_inds] - t).argmin()
print('time of {}, corresponds to vol index of {}'.format(t, vol_index))

#%%
# We will now view the parcel-wise COPEs over all timepoints within the trial, by outputting the parcel-wise COPEs as a 4D niftii object, where the 4th dimension is timepoint within trial.







