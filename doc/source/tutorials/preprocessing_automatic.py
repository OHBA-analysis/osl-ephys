"""
Automatic preprocessing using an osl-ephys config
===========================================

In this tutorial we will move the preprocessing pipeline that we built in the first tutorial to a osl-ephys `config`. The config is the most concise way to represent a full pipeline, and can be easily shared with other researchers. This tutorial looks as follows:

1. **The preprocessing config**
2. **Running a config with run_proc_chain**
3. **Adding custom functions to the pipeline**
4. **The preprocessing report**
5. **Optimizing your preprocessing config**
6. **Manually checking ICA**

"""



# We start by importing the relevant packages
import osl_ephys
import mne
import glob 
import yaml 
import os
import ipympl
from pprint import pprint

basedir = '/Users/matsvanes/osl_workshop/Preprocessing'#os.getcwd()
outdir = os.path.join(basedir, "preprocessed")
# generate the output directory if it doesn't exist
os.makedirs(outdir, exist_ok=True)

# There are multiple runs for each subject. We will first fetch all data using an osl-ephys utility
name = '{subj}_ses-meg_task-facerecognition_{run}_meg'
fullpath = os.path.join(basedir, 'data', name + '.fif')
datafiles = osl_ephys.utils.Study(fullpath)

# load the first run of subject 1.
sub1run1 = datafiles.get(subj='sub-01', run='run-01')[0]
raw = mne.io.read_raw_fif(sub1run1, preload=True)


#%%
# The preprocessing ``config``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The preprocessing config always has the same structure:
# 

config_text = """
  meta:
      event_codes:
          trigger1_description: trigger1_code
          trigger2_description: trigger2_code
          ...
  preproc:
      - method1: {setting: option, setting: option, ...}
      - method2: {setting: option, setting: option, ...}
    ...
"""

#%%
# We can specify the variable ``config_text`` in our Python script, or save the text itself (everything between the """ ... """) as a .yaml file. Whenever an osl-ephys function requires a config, it can be specified as either the path to the ``.yaml``-file, or the ``config_text`` variable.
# The `meta` field is currently only used for trigger information (`event_codes`), and is optional. It is used to give descriptic names to event triggers, and is used in combination with `mne.find_events`, epoching, etc. 
# 
# 
# In the `preproc` field, we specify each method we want to apply to the data, in the order in which we want to apply them (i.e., the methods will be applied serially). All methods from MNE-Python can be specified here, as well as some osl-ephys methods (see `osl_wrappers <https://osl-ephys.readthedocs.io/en/latest/autoapi/osl_ephys/preprocessing/osl_wrappers/index.html>`_). For each method we specify a dictionary with the settings; if we just use all default options, specify an empty dictionary ``{}``. 
# 
# To recap, this is the preprocessing pipeline we used in the previous tutorial:

# events
event_dict = {'famous/first': 5, 'famous/immediate': 6, 'famous/last': 7, 'unfamiliar/first': 13, 
              'unfamiliar/immediate': 14, 'unfamiliar/last': 15, 'scrambled/first': 17, 
              'scrambled/immediate': 18, 'scrambled/last': 19}
              
events = mne.find_events(raw, min_duration=0.005)

# channel types
raw.set_channel_types({"EEG061": "eog", "EEG062": "eog", "EEG063": "ecg"})

# filtering
raw = raw.filter(l_freq=0.25, h_freq=100)
raw = raw.notch_filter(freqs=(50, 100), picks='meg')

# bad segments
raw = osl_ephys.preprocessing.osl_wrappers.bad_segments(raw, picks='grad', segment_len=100)
raw = osl_ephys.preprocessing.osl_wrappers.bad_segments(raw, picks='mag', segment_len=100)

raw = osl_ephys.preprocessing.osl_wrappers.bad_channels(raw, picks='grad')
raw = osl_ephys.preprocessing.osl_wrappers.bad_channels(raw, picks='mag')

# ICA
ica = mne.preprocessing.ICA(n_components=64, random_state=42)
ica.fit(raw.copy().filter(l_freq=1, h_freq=None))

ecg_indices, ecg_scores = ica.find_bads_ecg(raw, ch_name='EEG063', method='correlation', threshold='auto')
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=['EEG061', 'EEG062'])

ica.exclude = []
ica.exclude += ecg_indices
ica.exclude += eog_indices

# optionally do some manual detection
raw = ica.apply(raw)

# epoching
epochs = mne.Epochs(raw, events, tmin=-0.5, tmax=1.5, event_id=event_dict)
epochs.drop_bad({"eog": 6e-4, "mag": 4e-11, "grad": 4e-10})




#%%
# Specifying the ``config`` 
# ^^^^^^^^^^^^^^^^^^
# Let's now try to put these processing steps into a config. We gave it a head start, but try to fill in the rest before continueing.
#
# :Tip: Use the  `Preprocessing API documentation <https://osl-ephys.readthedocs.io/en/latest/autoapi/osl_ephys/preprocessing/index.html>`_ and `MNE API <https://mne.tools/stable/api/python_reference.html>`_ to find the correct function names and settings. In the pipeline shown above, most steps are methods on the mne ``Raw`` class. You can see the available methods using ``dir(mne.io.Raw)``, and get help info as follows: e.g. ``help(mne.io.Raw.filter)``.
# 
# If a function is not a direct MNE method, check if it is an osl_wrapper (``dir(osl_ephys.preprocessing.osl_wrappers)``. Everything beginning with ``run_osl_`` can be used in the config (excluding the "run_osl_" prefix). Similarly, for an mne wrapper (``dir(osl_ephys.preprocessing.mne_wrappers)``), all functions that can be used in the config start with ``run_mne_``, but should be specified excluding the "run_mne_" prefix. You can use similar help commands to get more information on these functions (e.g., ``dir()``, ``help()``).
# 
# See how far you get. If you get stuck, find some extra tips below.

config_text= """
meta:
  event_codes:
    famous/first: 5
    famous/immediate: 6
preproc:
  - find_events:        {min_duration: 0.005}
  - set_channel_types:  {EEG061: eog, ...
"""

#%%
# :Tip: The ``bad_segments`` and ``bad_channels`` are osl_wrappers, e.g., see ``help(osl_ephys.preprocessing.osl_wrappers.bad_segments)`` for options.
#
# :Tip: Fitting ica and applying it to the data are mne_wrappers, as is the autoreject option. Check ``help(osl_ephys.preprocessing.mne_wrappers.ica_raw)``and ``help(osl_ephys.preprocessing.mne_wrappers.ica_autoreject)`` for the options.
# 
# Below you can find the full config. that we run ``ica_autoreject``, with ``apply: true``. This means that we run the automatic IC labeling, and remove those components from the data directly. We will come back to this later to find out how to select bad components manually.
# :note: we reduced the ``n_components`` in the ``ica_raw`` step to 20 to speed up processing.


config_text= """
meta:
  event_codes:
    famous/first: 5
    famous/immediate: 6
    famous/last: 7
    unfamiliar/first: 13
    unfamiliar/immediate: 14
    unfamiliar/last: 15
    scrambled/first: 17
    scrambled/immediate: 18
    scrambled/last: 19
preproc:
  - crop:               {tmin: 30}
  - find_events:        {min_duration: 0.005}
  - set_channel_types:  {EEG061: eog, EEG062: eog, EEG063: ecg}
  - filter:             {l_freq: 0.25, h_freq: 100}
  - notch_filter:       {freqs: 50 100}
  - bad_segments:       {segment_len: 100, picks: 'mag'}
  - bad_segments:       {segment_len: 100, picks: 'grad'}
  - bad_channels:       {picks: 'mag'}
  - bad_channels:       {picks: 'grad'}
  - ica_raw:            {n_components: 20, picks: 'meg', l_freq: 1}
  - ica_autoreject:     {apply: true}
  - epochs:             {tmin: -0.5, tmax: 1.5}
  - drop_bad:           {target: epochs, reject: {eog: 6e-4, mag: 4e-11, grad: 4e-10}}
"""


#%%
# Running a config with run_proc_chain
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Now we want to apply this preprocessing pipeline to the first file, we do this using the osl-ephys function ``osl_ephys.preprocessing.run_proc_chain(config, inputfile, ...)``. Let's first run this function, and then have a more detailed look at the in- and outputs to this funcction.


from osl_ephys.preprocessing import run_proc_chain

help(run_proc_chain)
dataset = run_proc_chain(config_text, raw, subject='sub001-ses01', outdir=outdir, overwrite=True)


print(f"run_proc_chain returned a dictionary with the following items: \n {dataset.keys()} \n\n")

#%%
# Let's dive into this function in a bit more detail. 
# There are two required inputs: 
#
# - ``config``: dict, path, or config text - as above
#
# - ``infile``: MNE object (e.g. Raw) or path to MEG data
# 
# The optional inputs are:
#
# - ``subject``: the subject/session specific identifier. This will be the name of the subdirectory in which derivative data are stored, as well as the prefix of the individual files.
#
# - ``ftype``: The extension for the preprocessed fif file, i.e., coming after the subject identifier (default ``preproc-raw``)
#
# - ``outdir``: The generic output directory in which the subdirectories will be created. By default the preprocessed data is not saved. Add a path here if you wish to save it.
#
# - ``logsdir``: The directory for processing and error logs. By default these are not saved.
#
# - ``reportdir``: Directory (see gen_report)
#
# - ``ret_dataset``: Return the dataset or not (not doing this only makes sense if you're saving the data to disk) - see below
#
# - ``gen_report``: osl-ephys can generate a report with summary measures and figures of the preprocessed data. We will have a closer look at this later.
#
# - ``overwrite``: Whether or not the overwrite existing data
#
# - ``skip_save``: List of dataset keys to skip writing to disk. If None, we don't skip any keys.
#
# - ``extra_funcs``: In case osl-ephys and MNE-Python don't have the function that you want to use, you can define the function yourself and specify the function name here
#
# - ``random_seed`` Random seed to set. If 'auto', a random seed will be generated. Random seeds are set for both Python and NumPy.
#
# - ``verbose``: print osl-ephys info
#
# - ``mneverbose``:  print MNE-Python info
# 
# 
# The ``dataset`` dictionary that is returned by ``run_proc_chain`` contains different items depending on the preprocessing pipeline. For example ``dataset["epochs"]`` is only returned because we specified our pipeline to include creating epochs. If we're saving the data to disk (i.e., ``outdir`` is specified), every item in ``dataset`` is saved seperately. The filenames will look something like:
#
# - ``raw``: `sub-001_run-01_preproc-raw.fif`
# - ``events``: `sub-001_run-01_events.npy` - numpy's way of saving data (load with `numpy.load()`)
# - ``epochs``: `sub-001_run-01_epo.fif`
# - ``event_id``: `sub-001_run-01_event_id.yml` - YAML file
# - ``ica``: `sub-001_run-01_ica.fif`
# 
# 
# Adding custom functions to the pipeline
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# osl-ephys has wrappers over most preprocessing functions from MNE-Python, plus a number of extra osl-ephys functions. However, you might want to use an algorithm that is not defined in either, for example from a third party toolbox, or a custom written function. You can easily create a wrapper such that you can use the function in the osl-ephys config. Just make sure that the function takes in the ``dataset`` as a first argument, and also returns the ``dataset```. You function should manipulate any of the keys in ``dataset``, and optionally, return a new key. New keys are by default saved as .pkl files in the subject directory (i.e., as ``key_name.pkl``). You can also add a figure to ``dataset``, which will be saved as .png (e.g., "key_name.png"). 
#
# Let's see what adding a custom function to the config looks like in practice.

def custom_function(dataset, option1=None):
    # this is the main body of the function
    return dataset

#%%
# We add the following line to our ``config``:
"""
- custom_function:        {option1: true}
"""

# %% 
# and the following input to ``run_proc_chain```:
#
extra_funcs=[custom_function]

#%% 
# We'll now run the preprocessing pipeline again, but we'll now save our data to disk, generate a report, and include an extra function (``ica_kurtosisreject``), which marks ICs as bad if the kurtosis passes a certain threshold, and removes them from the data. This function is similar to ``ica_autoreject``, which uses the correlation between the ICs and EOG/ECG channels to mark components as bad. Both these functions have the option to remove the bad components from the data.


def ica_kurtosisreject(dataset, userargs):
    import numpy as np
    from scipy.stats import kurtosis
    threshold = userargs.get('threshold', 10)
    apply = userargs.get('apply', True) # whether or not to remove the rejected components from the data
    
    # Since osl-ephys returns information on which processing stage is currently running, 
    # we might want to print out some useful information about our custom function.
    print('osl-ephys Stage - {0}'.format('ICA Kurtosis Reject'))
    print('userargs: {0}'.format(str(userargs)))
    
    
    # The main body of the function deals with actually computing the kurtosis on the
    # IC timecourses, and in turn, marks the components that surpass the threshold 
    # as bad, e.g. by adding their indices to dataset['ica'].exclude.
    raw=dataset['raw']
    ica=dataset['ica']
    ic_map = ica.get_components()[mne.pick_types(ica.info, meg=True, eeg=False), :] # IC timecourses (sensors x number of ICs)
    ic_timeseries = np.transpose(
            np.matmul(np.transpose(ic_map), raw.get_data()[mne.pick_types(ica.info, meg=True, eeg=False), :])) # get the IC timeseries by multiplying the IC map with the Raw timeseries.
            
    k = kurtosis(ic_timeseries, fisher=False) # compute the kurtosis
    bad_components = np.where(k>threshold)[0] # find the components with a kurtosis value larger than the threshold
    if len(bad_components>0):
        dataset['ica'].exclude.extend(bad_components) # add these indices to ica.exclude.
        
    if apply:
        dataset['ica'].apply(dataset['raw'])
    return dataset


config_text= """
meta:
  event_codes:
    famous/first: 5
    famous/immediate: 6
    famous/last: 7
    unfamiliar/first: 13
    unfamiliar/immediate: 14
    unfamiliar/last: 15
    scrambled/first: 17
    scrambled/immediate: 18
    scrambled/last: 19
preproc:
  - find_events:        {min_duration: 0.005}
  - set_channel_types:  {EEG061: eog, EEG062: eog, EEG063: ecg}
  - filter:             {l_freq: 0.25, h_freq: 100}
  - notch_filter:       {freqs: 50 100}
  - bad_segments:       {segment_len: 1000, picks: 'mag'}
  - bad_segments:       {segment_len: 100, picks: 'grad'}
  - bad_channels:       {picks: 'mag'}
  - bad_channels:       {picks: 'grad'}
  - ica_raw:            {n_components: 20, picks: 'meg', l_freq: 1}
  - ica_autoreject:     {apply: true}
  - ica_kurtosisreject: {threshold: 10, apply: true}
  - epochs:             {tmin: -0.5, tmax: 1.5}
  - drop_bad:           {target: epochs, reject: {eog: 6e-4, mag: 4e-11, grad: 4e-10}}
"""

run_proc_chain(config_text, sub1run1, subject= 'sub001-ses01', outdir=outdir, ret_dataset=False, gen_report=True, 
                                 overwrite=True, extra_funcs=[ica_kurtosisreject])

#%%
# As an alternative to running `run_proc_chain` from within Python, osl-ephys allows it to be run from the terminal's command line as well. The above command would then look as follows:
#
# ``osl_preproc my_config.yml list_of_raw_files.txt --outdir /path/to/my/output_dir --overwrite``
#
#
# Note that this currently is not possible when using custom functions.
# 
# 
# The preprocessing report
# ^^^^^^^^^^^^^^^^^^^^^^^^
# The preprocessing report generates a folder for each MEG file that contains all the figures that are generated for this file. This is all collected in the ``subject_report.html``' HTML file. This allows you to browse through your files for different quality metrics. In addition, there is a ``summary_report.html``, which contains summary metrics that can guide you to look at individual datasets in the `subject_report` (for example, when one dataset has a lot of bad channels).
# The idea of the report is to help you guide optimizing your preprocessing pipeline, and checking data quality. If a researcher asks you how the quality of the data is, the question is not trivial. With the report, we hope to give you a tool that quantifies some important metrics. It is not exhaustive, so if there's missing anything, please `open an issue on GitHub <https://github.com/OHBA-analysis/osl-ephys/issues>`_. 
# 
# 
# Now open the report (you find it in your ``preproc_report`` directory). We will run through the report step by step.
# Navigating through the report can be done through mouse clicks or using the arrows on your keyboard (up-down for different tabs; left/right for different files).
# 
# - Info: Contains meta data. Filenames, data size, and how many channels and events are in the data.
#
# - Time Series: Reports how much of the data was annotated as bad segments. Then shows the variance for each channel type (bad segments are highlighted in red), including and excluding outliers. Below you will see the raw data from the EOG and ECG channels.
#
# - Channels: Shows a histogram of the per sensor variance including and excluding outliers. Below it lists which sensors are labeled as "bad" (also annotated in red)
#
# - Power Spectra: The power spectrum seperately for each channel type; full spectrum (left) and zoomed in (right). Each line is a sensor.
#
# - Digitisation: Shows the Polhemus and HPI information.
#
# - ICA: For each component labeled as "bad", it shows the topography for each sensor type (top left), the power spectrum (bottom left), the ERP (top right - no ERP in our case because ICA was applied before epoching), and the variance (bottom right).
#
# - Logs: a detailed log of all the processing applied to the dataset.

#%% 
# Optimizing your preprocessing config
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The report is one tool that can help you to optimize your preprocessing, but one other tool you should keep in mind is visualizing the data itself (e.g., using the databrowser ``raw.plot()``).
# In the **Time Series** tab we see for example that there are still points in time with high variance, both in magnetometers and gradiometers. Let's see if we can clean up the data a bit more by adding extra bad segment detection, based on the temporal derivative (i.e., how strong the signal changes; using ``mode: 'diff'```). Let's also try to remove the segment in the EOG where we see a big amplitude increase.  
# 
# Note that here, we drop the custom function for ICA labelling that we used before. We will also set ``apply: false`` in ``ica_autoreject```, so we can manually check the components labeled as bad before we removed them from the data. Therefore, we will also leave out the epoching, since we want to apply this after we're done with ICA. 


config_text= """
meta:
  event_codes:
    famous/first: 5
    famous/immediate: 6
    famous/last: 7
    unfamiliar/first: 13
    unfamiliar/immediate: 14
    unfamiliar/last: 15
    scrambled/first: 17
    scrambled/immediate: 18
    scrambled/last: 19
preproc:
  - crop:               {tmin: 30}
  - find_events:        {min_duration: 0.005}
  - set_channel_types:  {EEG061: eog, EEG062: eog, EEG063: ecg}
  - filter:             {l_freq: 0.25, h_freq: 100}
  - notch_filter:       {freqs: 50 100}
  - bad_segments:       {segment_len: 100, picks: 'mag'}
  - bad_segments:       {segment_len: 100, picks: 'grad'}
  - bad_segments:       {segment_len: 500, picks: 'mag', mode: 'diff'}
  - bad_segments:       {segment_len: 500, picks: 'grad', mode: 'diff'}
  - bad_segments:       {segment_len: 1000, picks: 'eog'}
  - bad_channels:       {picks: 'mag'}
  - bad_channels:       {picks: 'grad'}
  - ica_raw:            {n_components: 20, picks: 'meg', l_freq: 1}
  - ica_autoreject:     {apply: false}
"""

# We'll save the config for later use.
config = yaml.safe_load(config_text)
with open('config.yaml', 'w') as file:
    yaml.dump(config, file)

run_proc_chain(config_text, sub1run1, subject= 'sub001-ses01', outdir=outdir, ret_dataset=False, gen_report=True, 
                                 overwrite=True)


#%%
# Manually checking ICA
# ^^^^^^^^^^^^^^^^^^^^^
# Now that we are happy with the preprocessing pipeline on this dataset, we could load in the data, make manual adjustments to the IC's that were labeled as bad (or at least check them), remove those from the data and save the clean data. We also might want to re-run the report with the updated ICA information.
# In the previous tutorial we used an ICA plotting tool, but it doesn't work in each IDE. We also have a command line function for this, called ``osl_ica_label``. You can use this from the terminal (make sure you're in an ``osle`` environment.)
# This function requires a few inputs, firstly you need to specify what to do with the components marked as bad. Options are: None (only save the ICA object, don't remove any from the M/EEG data), "manual" (only remove the manually labeled components; particularly useful if you've already removed automatically detected components), "all" (remove all labeled, automatic and manual, from the MEEG data).
# The next input argument is the general output directory, and third, the subdirectory name of the subject. For example:
# 
# ``(osle) > osl_ica_label None preprocessed sub001-ses01``



#%%
# When we close the figure, the ICA object is automatically saved. The log and report are also updated, and if we specified to remove components from the M/EEG data, this will also have been carried out.
# If we haven't yet removed the components from the data, we can do so post-hoc using another command line tool:
#
# ``(osle) > osl_ica_reject preprocessed sub001-ses01``

