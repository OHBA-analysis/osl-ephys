
"""
Manual preprocessing in Python with MNE-Python and osl-ephys
======================================================

In this tutorial we will build a preprocessing pipeline using functions from MNE-Python and osl-ephys. Both packages can be used for MEG and EEG data analysis, irrespective of the manufacturer of the recording equipment. 
From our experience, preprocessing pipelines are rarely directly generalizable between different datasets. This can be due to different study designs, different sources of noise, different study populations, etc. As such, we recommend to always interact with your data initially. 
This can for example be done by applying preprocessing steps one by one and visualising the data each time to see what effects every step has on your data. In this section we will do just that; starting from the raw data.

**Note**: the data we'll use has already had a MaxFilter applied to it. MaxFilter is Elekta licensed software, and is also only needed for Elekta/Megin data. It is used to remove external noise (e.g., environmental noise) and do head movement compensation. 
Maxfilter uses some extra reference sensors in the MEG together with Signal Space Seperation (SSS) to achieve this. MaxFilter has various settings, which we will not go into here, but osl-ephys does have a `wrapper <https://osl-ephys.readthedocs.io/en/latest/autoapi/osl_ephys/maxfilter/maxfilter/index.html>`_ for the 
Elekta software with some explanations of settings. Furthermore, `MNE-Python also has a maxfilter that doesn't require a license <https://mne.tools/stable/generated/mne.preprocessing.maxwell_filter.html>`_. Besides these references, also have a look at the 
`MaxFilter user manual <https://ohba-analysis.github.io/osl-docs/downloads/maxfilter_user_guide.pdf>`_ and at `these guidelines <https://lsr-wiki-01.mrc-cbu.cam.ac.uk/meg/maxpreproc>`_).


In this tutorial, we will start from a typical pipeline that has shown to be a good first pass in many datasets, and adapt it to the current dataset.
Also see the `MNE-Python preprocessing tutorial <https://mne.tools/stable/auto_tutorials/preprocessing/index.html>`_.

Building a preprocessing pipeline with functions from MNE-Python and osl-ephys.

1. **M/EEG data in MNE-Python**
    1. Getting the data
    2. Visualizing the data
2. **Preprocessing**
    1. Filtering: band-pass and notch filtering
    2. Automated bad segment/channel detection
    3. Removing cardiac and occular artifacts with ICA
3. **Creating Epochs**
4. **Concluding remarks**


Download the dataset
====================
We will download example data hosted on `OSF <https://osf.io/zxb6c/>`_

"""


import os
import mne
import osl_ephys
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

basedir = os.getcwd()

def get_data(name):
    """Download a dataset from OSF."""
    if os.path.exists(name):
        return f"{name} already downloaded. Skipping.."
    os.system(f"osf -p zxb6c fetch '1. Preprocessing/{name}.zip'")
    os.system(f"unzip -o {name}.zip")
    os.remove(f"{name}.zip")
    return f"Data downloaded to: {name}"

# Download the dataset (~2.2 GB)
get_data(os.path.join(basedir, "data"))


# prepare output directory
outdir = os.path.join(basedir, "preprocessed")
# generate the output directory if it doesn't exist
os.makedirs(outdir, exist_ok=True)

#%%
# Loading the data
# ^^^^^^^^^^^^^^^^
# The original data contains multiple runs for each subject. We will first fetch all data using the osl-ephys ``Study`` utility.  study contains all files that match the pattern, where each '{...}' is replaced by a wildcard. 
# We can use the ``get`` method to get a list of all matching files, optionally filtered by either of the wildcards. 
study = osl_ephys.utils.Study(os.path.join(basedir, 'data', '{subj}_ses-meg_task-facerecognition_{run}_meg.fif'))

# view a list of all matching files:
all_files = study.get()
print('Found {} files'.format(len(all_files)))
pprint(all_files)

# Get the first run of subject 1, which we'll use in this tutorial
sub1run1 = study.get(subj='sub-01', run='run-01')[0]
pprint(sub1run1)

#%% 
# Visualizing the data
# ^^^^^^^^^^^^^^^^^^^^
# We will first load a single dataset. MNE-Python has different classes for handling data at different processing stages. 
# The classes that are most used in sensor space analyses are ``Raw``, ``Epochs``, ``Evoked``, as well as some classes for (time-) frequency data. Preprocessing will typically be done on raw, continuous data (e.g. on the ``Raw`` class). More info can be found `here <https://mne.tools/stable/api/most_used_classes.html>`_.
# When we load in the data, some information about the data will be printed, like the full duration of the data, the number of channels that are present, and the full size. We can get more details by looking at raw.info, which is a Python dictionary. Lastly, we can get the full data matrix with ``raw.get_data()`` to see its shape (channels by time), or to directly manipulate the data.

# Load a single dataset
raw = mne.io.read_raw_fif(sub1run1, preload=True)
print(raw)
print(raw.info)
print(raw.get_data().shape)

#%%
# We can detect events using mne.find_events. Each trigger code is associated to a condition:
#
# - 5: famous face - first presentation
# - 6: famous face- immediate repetition
# - 7: famous face - last repetition
# - 13: unfamiliar face - first presentation
# - 14: unfamiliar face - immediate repetition
# - 15: unfamiliar face - last repetition
# - 17: scrambled face - first presentation
# - 18: scrambled face - immediate repetition
# - 19: scrambled face - last repetition
# 
# We'll ignore all events that are too short.

# Detect events
events = mne.find_events(raw, min_duration=0.005)
event_color = {}
event_dict = {'famous/first': 5, 'famous/immediate': 6, 'famous/last': 7, 'unfamiliar/first': 13, 
              'unfamiliar/immediate': 14, 'unfamiliar/last': 15, 'scrambled/first': 17, 
              'scrambled/immediate': 18, 'scrambled/last': 19}

fig, ax = plt.subplots(1,1, figsize=(8,6))
fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], event_id=event_dict, on_missing='ignore', verbose='error', axes=ax)

#%%
# Let's now visualize the data. We can browse through the channels and time using the arrow key (you may need to click on the figure first). 
# Before we plot the data, we set the EOG and ECG channels so we don't have to remember this.


raw.set_channel_types({"EEG061": "eog", "EEG062": "eog", "EEG063": "ecg"})
fig = raw.plot(n_channels=20)
fig.set_size_inches(8,8)

#%% 
# Preprocessing
# ^^^^^^^^^^^^^
# It's important to keep in mind what our goal is in preprocessing the data. We want to remove artefacts or other 
# sources of variance that are not of interest to us (for example because they are related to the environment). 
# Generally, we refer to these sources of variance as "noise". This is challenging because it's not always clear 
# which parts of the data are noise and which are not. If we preprocess our data too rigorously, we might risk 
# throwing out the baby with the bathwater. For this reason, we are cautious and interact with the data when we 
# develop our preprocessing pipeline. Some of the things that are useful to look at for checking the data quality 
# is the variance of the data (over time, and over channels), the time domain signal traces, and the frequency 
# domain power spectral density (PSD, or power).

# Let's create a function with which we can easily look at the variance of the data
def plot_var(raw):
    """Plot the variance of the data over time and channels."""
    mag = raw.get_data(picks='mag', reject_by_annotation='NaN')
    grad = raw.get_data(picks='grad', reject_by_annotation='NaN')

    fig, ax = plt.subplots(2,2)
    plt.axes(ax[0,0])
    plt.plot(raw.times, np.nanvar(grad, axis=0)), plt.title('Variance over time (GRAD)'),  plt.xlabel('Time (s)'), plt.ylabel('Variance (T/m)')
    plt.axes(ax[1,0])
    plt.plot(raw.times, np.nanvar(mag, axis=0)), plt.title('Variance over time (MAG)'), plt.xlabel('Time (s)'), plt.ylabel('Variance (T)')


    im,_ = mne.viz.plot_topomap(np.nanvar(grad, axis=1), raw.copy().pick('grad').info, axes=ax[0,1], show=False)
    ax[0,1].set_title('Variance over gradiometers')
    cax =  ax[0,1].inset_axes([1.1, 0, 0.05, 0.8])
    plt.colorbar(im, cax=cax, label='Variance (T/m)', orientation='vertical')
    
    im,_ = mne.viz.plot_topomap(np.nanvar(mag, axis=1), raw.copy().pick('mag').info, axes=ax[1,1], show=False)
    ax[1,1].set_title('Variance over magnetometers')
    cax =  ax[1,1].inset_axes([1.1, 0, 0.05, 0.8])
    plt.colorbar(im, cax=cax, label='Variance (T/m)', orientation='vertical')
    
    plt.tight_layout()
    plt.show()
    return fig, ax 

#%%
# Now plot the variance over time and over channels - seperately for each channel type. Note that the temporal variance peaks at ~20 s and then suddenly increases around 150 s.

# Plot variance over time and over channels
fig, ax = plot_var(raw)

#%%
# We'll also plot the power spectrum using MNE-Python functions. Note that we are already working with downsampled data (250 Hz), so the maximum frequency currently is 125 Hz.

psd = raw.compute_psd(picks='meg')
fig, ax = plt.subplots(2,1, figsize = (8,6))
fig = psd.plot(axes=ax)
plt.suptitle('Note the peaks at 50 Hz and 100 Hz in the plots - this corresponds to line noise')

#%%
# Filtering: band-pass and notch filtering
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Some preprocessing options are relatively standard, because they deal with artefacts that are always expected. 
# One such artefact is line noise (originating from the A/C output at 50 Hz in Europe, or 60 Hz in the USA), as can be seen in the power spectrum above.
# We can remove this with a filter (i.e. a notch filter). Another artefact that can easily be removed using filters is high frequency noise (low pass filter).
# Note that the filter cut off depends on your sampling frequency, and the analyses you intend to do and the hypotheses you have.
# It is good practice to filter your data before downsampling, because doing it the other way around can introduce `aliasing issues <https://en.wikipedia.org/wiki/Anti-aliasing_filter>`_.
# The reason our data is already downsampled is purely for practical reasons - usually we would do this at a later stage.
# Let's band-pass filter the signal and remove line noise, looking at the effect on the power spectrum every time.
#
# :note: The method ``raw.compute_psd()`` returns MNE-Python's Spectrum class. See more `here <https://mne.tools/stable/generated/mne.time_frequency.Spectrum.html>`_.
#
# We start with the band-pass (BP) filter (a combination of a low- and high-pass filter).
# The high pass filter is used to remove slow drifts in the signal, which are often present in M/EEG data.
# The low pass filter is used to remove high frequency noise (like muscle activity).
#
# :warning: In Python, if we assign a variable to an existing one, like ``raw_new = raw```, the two variables stay linked and the data is not actually copied. This means that if we manipulate ``raw_new```, it will also manipulate ``raw``. Thus, we should explicitly copy ``raw`` as follows: ``raw_new = raw.copy()``.


psd = raw.compute_psd(picks='meg')
fig, ax = plt.subplots(2,2, figsize = (10,6))
psd.plot(axes=ax[:,0])
ax[0,0].set_title('Raw data \n Gradiometers (204 channels)')

raw_bp = raw.copy().filter(l_freq=0.25, h_freq=100)
psd_bp = raw_bp.compute_psd(picks='meg')

psd_bp.plot(axes=ax[:,1])
ax[0,1].set_title('After band-pass (BP) filter \n Gradiometers (204 channels)')

#%%
# We can clearly see the spectral power steeply decrease above 100 Hz. It's not as easy to see the effect of the high-pass filter in the power spectrum plot (we would if we zoomed in to the 0-1 Hz range), but it is very clear if we look at the time domain signal. Here we just visualize the entire data from a single channel, before and after BP filtering.

fig, ax = plt.subplots(1,2)
ax[0].plot(raw.times[:], raw.get_data()[0,:])
ax[0].set_xlabel('Time (s)')
ax[0].set_title('There is a slow drift \n in the raw signal')
ax[1].plot(raw.times[:], raw_bp.get_data()[0,:])
ax[1].set_xlabel('Time (s)')
ax[1].set_title('The slow drift is removed \n after BP-filtering')
plt.show()

#%%
# Now use a notch filter and plot again. (Note that we first copy the raw data so that we keep an original copy).

freqs = (50, 100)
raw_notch = raw_bp.copy().notch_filter(freqs=freqs, picks='meg')
psd_notch = raw_notch.compute_psd(picks='meg')

# Plot the previous two figures again
fig, ax = plt.subplots(2,3, figsize = (10,6))
psd.plot(axes=ax[:,0])
ax[0,0].set_title('Raw data \n Gradiometers (204 channels)')
psd_bp.plot(axes=ax[:,1])
ax[0,1].set_title('After band-pass (BP) filter \n Gradiometers (204 channels)')

# Plot the BP + notch filtered PSD
psd_notch.plot(axes=ax[:,2]) # See the plot above
ax[0,2].set_title('After BP and notch filter \n Gradiometers (204 channels)')

# Make sure the y-axes are the same, to ease comparison
[ax[0,i].set_ylim((0,30)) for i in range(3)]
[ax[1,i].set_ylim((0,70)) for i in range(3)]

#%%
# We can see that the notch filter did a good job in removing the line noise. By this stage we have already manipulated the data quite a bit. As mentioned when we set out with this tutorial, the goal of preprocessing is to remove variance from the data in which we are not interested. Let's have a look at the variance of our data so far. The spatial, and temporal variance have changed quite dramatically!

fig, ax = plot_var(raw_notch)

#%%
# Automated bad segment/channel detection
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Even after filtering there can be large artefacts left in the data, for example resulting from head and eye movements, muscle twitches, and other (unknown) physiological sources. Hence, we next perform bad segment/channel detection. This can be done manually, i.e., by going through the data and manually selecting bad segments/channels (e.g., the gradiometer segment right after the start of the recording in the top left plot above). Alternatively, we can use automatic detection using osl-ephys tools (``osl_ephys.preprocessing.osl_wrappers.bad_segments``, ``osl_ephys.preprocessing.osl_wrappers.bad_channels``). These tools use a `Generalized ESD test (Rosner, 1983) <https://www.jstor.org/stable/1268549>`_ - a procedure for removing outliers in univariate data that approximately follows a normal distribution. 

#%%
# Bad segments
# ************
raw_badseg = osl_ephys.preprocessing.osl_wrappers.bad_segments(raw_notch.copy(), picks='grad')
raw_badseg = osl_ephys.preprocessing.osl_wrappers.bad_segments(raw_badseg, picks='mag')
fig, ax = plot_var(raw_badseg)

#%%
# In the bad segment detection above we used the default parameters. This is a good start: it has removed the high variance event in gradiometers at the end of the recording (see top left plot), but there are still some time points with high variance. This illustrates that different datasets and artefact types might require different settings, or even running bad segment detection multiple times with different settings. 
# By default, bad segment detection is run on 1000 sample segments, and with a significance level of 0.05. 
# :Excercise Try running the bad segment detection with different settings. For example, try a smaller segment length (e.g., 50, 100, 500, 2000 samples) and a higher significance level (e.g., 0.1). Play around with this until you are satisfied with the results before continueing (i.e., the variance looks equally distributed over time). Often it can also help to run the bad segment detection on the temporal derivative of the data (use mode='diff'), or on the variance or kurtosis, rather than the standard deviation (metric=...)):


#%%
# We found that using a segment length of 100, and a significance of 0.05 does a reasonable job. Keep in mind that there is not neccessarily one correct answer, and in any case we'll have to see how well these settings generalise to other datasets later. Doing these steps on multiple datasets will guide us to find the best general settings for this dataset.
raw_badseg = osl_ephys.preprocessing.osl_wrappers.bad_segments(raw_notch.copy(), picks='grad', segment_len=100)
raw_badseg = osl_ephys.preprocessing.osl_wrappers.bad_segments(raw_badseg, picks='mag', segment_len=100)

#%%
# The variance looks a lot more equally distributed over time now. Next, let's do the bad channel detection. In the variance topography plots we can see that the variance range is small and there are no clear outliers. In other datasets there might be, so we'll build bad channel detection into our pipeline anyway.

#%%
# Bad channels
# ************
raw_badchan = osl_ephys.preprocessing.osl_wrappers.bad_channels(raw_badseg.copy(), picks='grad')
raw_badchan = osl_ephys.preprocessing.osl_wrappers.bad_channels(raw_badchan, picks='mag')
print(f"These channels were marked as bad: {raw_badchan.info['bads']}")
fig, ax = plot_var(raw_badchan)

#%%
# The bad channel detection still has marked a few channels as bad. Usually the amount of bad channels in MEG data is quite limited, because we don't expect individual channels to misbehave. This is different in EEG, where the conductance of certain channels might be particularly bad.
# Let's visualize the data again. The segments that we detected before are annotated as bad. This means they are not removed from the data, but an annotation is saved as meta info. Further MNE/osl-ephys-functions have different ways of handling this, e.g. by replacing those segments with NaN's, omitting the data, etc. In the plot below, the bad segments are annotated in red, bad channels are gray.
# We can interact with this figure for manually annotating segments (draggin a window over a time period) or channels (clicking on a channel).
    
fig = raw_badchan.plot(duration=100, n_channels=50)

#%%
# Removing cardiac and occular artifacts with ICA
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# As we can see in the data browser, there are still some sources of high variance present in the data. Channel MEG0143 has a strong rhythmic spiking present, and there are other high variance transient events. The former is most likely due to cardiac activity (the heart contains an electrical pacemaker, and is also a muscle), which has a strong effect on the MEG signal (less so in EEG). Since this is a regular signal that is present in many channels, removing it with bad segment/channel detection is unfeasible. Instead, Independent Component Analysis is a common technique for removing this type of noise.
# Similarly, eye blink and saccades have a big influence on the MEG (and EEG) data. Again due to eye muscle activity, but also because the eye itself is polarized (the cornea is net positive, and the retina net negative), and thus moving the eye changes the magnetic field.
# When running ICA, it is recommended to have a high pass filter beforehand, because ICA doesn't work well when there are slow drifts in the signal. A 1 Hz high pass is thus used below (see here fore more info). Further, we have to specify the amount of components. MaxFilter effectively reduces the rank of our data from 306 (i.e. all MEG channels) to about ~70 (when we used the default options). It doesn't make sense to look at more components than that.
# We can fit ICA using the following two lines (This takes a couple of minutes to run though (of course ``random_state=42`` is not essential!).

ica = mne.preprocessing.ICA(n_components=64, random_state=42)
ica.fit(raw_badchan.copy().filter(l_freq=1, h_freq=None))

#%%
# Alternatively, we can load the precomputed ICA object.

ica = mne.preprocessing.read_ica('ica.fif')

#%%
# We now have to label the components that we think correspond to cardiac/occular activity. Generally, there are two strategies to use here (alse see `MNE-Python's tutorial on this <https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#sphx-glr-auto-tutorials-preprocessing-40-artifact-correction-ica-py>`_):
#
# 1. Correlating component time courses with the recorded ECG and EOG data. Those components that have a high correlation likely correspond to these types of noise.
# 2. Visualizing the component time course and spatial topography and use our knowledge of the biophysics of these signals to manually detect components corresponding to these types of noise.
# 
# We recommend a combination of the above. The first option tends to give a good first pass, but it sometimes misses components. This is especially detrimental if it does a better job in one subject group versus another (e.g. healthy population vs. patient group), or when the EOG/ECG recording is missing or of bad quality.
# We will first use the correlation with ECG and EOG to find (potential) artifact related ICs. We'll then go through the components manually to see whether the automatic detection was accuracte and sufficient.


# Correlating component time courses with the recorded ECG and EOG data
ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='EEG063', method='correlation', threshold='auto')
print(ecg_indices)

#%%
# The automatic detection found 2 components that are likely to be related to the ECG. It is typical in MEG data to have 1-3 components related to the ECG, and 1-2 components related to EOG. Note that this is only a rule of thumb. In EEG data, it is not uncommon to not find any components related to the ECG.
# Add the components indices to ica.exclude
ica.exclude = ecg_indices

#%%
# Now we do the equivalent for occular artifacts.

eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['EEG061', 'EEG062'])
print(eog_indices)

# Add these to ica.exclude
ica.exclude += eog_indices

print(ica.exclude)

#%%
# Let's use osl-ephys's ICA databrowser to make corrections where needed. The browser will show the topographies on the left (seperate for each channel type), and the time course on the right. We can click on a time course if we want to label a component as bad (another click unlabels the component). After clicking, we can optionally use numbers 1-5 to specify what type of artefact we're labeling. This is currently not used for anything, but can aid later analyses of ICA (it is saved in ``ica.labels_``).
#
# :note: Interacting with the figure in Jupyter Notebook does not work well. In the `preprocessing using the osl-ephys config API tutorial <https://osl-ephys.readthedocs.io/en/latest/tutorials_build/preprocessing_automatic.html#manually-checking-ica>_` we'll show a way to do this using a command line function. Also see `How do I select which components to remove in ICA <https://osl-ephys.readthedocs.io/en/latest/faq.html#how-do-i-select-which-components-to-remove-in-ica>`_. For now, we will create a simple python script that will be launched from the terminal. We may have to browse through time/space for the figure to render correctly.
#
# When we're done, we can close the window. ica.exclude is then updated and saved. We will load the edited ICA data, and then remove the identified bad components from the data using ``ica.apply()``.
#
# :note: The components are only removed from the data after calling ``ica.apply(raw)``. When we are happy with our preprocessing and are ready to save the clean data, we can do so with ``clean.save(filepath)`` (see `here <https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.save>`_)

# Use osl-ephys's ICA databrowser to make corrections where needed. We'll write a python script for this and execute it from the terminal.
# temporarily save the raw and ica data
fname_data = sub1run1.replace('data', 'preprocessed').replace('.fif', '_preproc_raw.fif')
fname_ica = sub1run1.replace('data', 'preprocessed').replace('.fif', '_ica.fif')
raw.save(fname_data)
ica.save(fname_ica)

import os
with open("plot_ica_from_terminal.py", "w") as f:
    f.write(f"""\
        import mne
        from osl_ephys.preprocessing.plot_ica import plot_ica
        ica = mne.preprocessing.read_ica('{fname_ica}')
        raw = mne.io.read_raw_fif('{fname_data}', preload=True)
        fig = plot_ica(ica, raw)
        ica.save('{fname_ica}', overwrite=True)
        """)
    
# Run the script from the terminal
import subprocess
proc = subprocess.Popen('python plot_ica_from_terminal.py', stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
output, _ = proc.communicate()
    
# Once we're done with the interactive figure, we can close it. This will save the ICA object with the updated exclude list, which we will reload here.
ica = mne.preprocessing.read_ica(fname_ica)
print(f'The following components were labeled as bad: {ica.exclude}')
print(f'These are the contents of ica.labels_: {ica.labels_}')

# Remove bad components from the data
clean = ica.apply(raw_badchan.copy())

# Save the clean data
clean.save(fname_data, overwrite=True)

#%%
# Creating Epochs
# ^^^^^^^^^^^^^^^
# We now have clean, continuous data. We've already looked for events earlier, which we can now use to epoch our data, using MNE-Python's Epochs class. This creates epochs for all events, running from 0.5 seconds before till 1.5 seconds after the event. For more info on the Epochs class, see `here <https://mne.tools/stable/auto_tutorials/epochs/10_epochs_overview.html>`_.

epochs = mne.Epochs(clean, events, tmin=-0.5, tmax=1.5, event_id=event_dict)
print(epochs)
print(f"epochs has the following size [epoch x channel x time]: {epochs.get_data().shape}")

#%%
# We can select different events with epochs["event_name"]. So for example epochs["famous/first"], but conveniently, we can also select all famous events at once:

epochs['famous']

#%%
# Lastly, we'll remove epochs with particularly high peak-to-peak amplitudes, as this indicates there might still be segments in the data with high variance, that we didn't find earlier. We also include EOG peak-to-peak amplitude, as high amplitudes indicate sacades.
# Note that we have to specify the peak-to-peak amplitude in SI units for each channel type

epochs.drop_bad({"eog": 6e-4, "mag": 4e-11, "grad": 4e-10}) # units: V, T, T/m 

fname_epochs = sub1run1.replace('data', 'preprocessed').replace('.fif', '_epo.fif')
epochs.save(fname_epochs)

#%%
# Concluding remarks
# ^^^^^^^^^^^^^^^^^^
# In this tutorial we have built a preprocessing pipeline by manipulating the data step by step. We have used quite a few techniques for cleaning up the data, but note that this is not exhaustive. For example, if we expect some bad channels in every subject (as in EEG), we might want to interpolate bad channels. Or, you might have artifacts that are specific to your environment (e.g. interference from air conditioning) or study population (e.g. subjects containing metal artefacts). Thus, always think about the sources of noise you expect in your data, and what preprocessing option this requires. Then, keep interacting with your data to find a pipeline that cleans up the data satisfactorily.
# In the next tutorial, we will make a config dictionary that contains all these preprocessing steps in one place, and then apply all steps in sequence using a single function call to osl-ephys.