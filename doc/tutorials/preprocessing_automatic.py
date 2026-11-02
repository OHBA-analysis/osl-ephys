"""
Automatic preprocessing using an osl-ephys config
=================================================

In this tutorial we will move the preprocessing pipeline that we built in the first tutorial to an osl-ephys ``config``. The config is the most concise way to represent a full pipeline, and can be easily shared with other researchers. This tutorial looks as follows:

1. **The preprocessing config**
2. **Running a config with run_chain**
3. **Adding custom functions to the pipeline**
4. **The QC report**
5. **Optimizing your preprocessing config**
6. **Manually checking ICA**
"""

# We start by importing the relevant packages
import os
import yaml
from pprint import pprint

import mne
import osl_ephys
from osl_ephys.pipeline import run_chain

basedir = os.getcwd()
outdir = os.path.join(basedir, "preprocessed")

# There are multiple runs for each subject. We will first fetch all data using an osl-ephys utility
name = '{subj}_ses-meg_task-facerecognition_{run}_meg'
fullpath = os.path.join(basedir, 'data', name + '.fif')
datafiles = osl_ephys.io.Study(fullpath)

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
# In the `preproc` field, we specify each method we want to apply to the data, in the order in which we want to apply them (i.e., the methods will be applied serially). All methods from MNE-Python can be specified here, as well as some osl-ephys methods (see `osl_wrappers <https://osl-ephys.readthedocs.io/en/latest/autoapi/osl_ephys/preproc/osl_wrappers/index.html>`_). For each method we specify a dictionary with the settings; if we just use all default options, specify an empty dictionary ``{}``. 
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
raw = osl_ephys.preproc.detect_bad_segments(raw, picks='grad', segment_len=100)
raw = osl_ephys.preproc.detect_bad_segments(raw, picks='mag', segment_len=100)

raw = osl_ephys.preproc.detect_bad_channels(raw, picks='grad')
raw = osl_ephys.preproc.detect_bad_channels(raw, picks='mag')

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
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's now try to put these processing steps into a config. We gave it a head start, but try to fill in the rest before continuing.
#
# :Tip: Use the  `Preprocessing API documentation <https://osl-ephys.readthedocs.io/en/latest/autoapi/osl_ephys/preproc/index.html>`_ and `MNE API <https://mne.tools/stable/api/python_reference.html>`_ to find the correct function names and settings. In the pipeline shown above, most steps are methods on the mne ``Raw`` class. You can see the available methods using ``dir(mne.io.Raw)``, and get help info as follows: e.g. ``help(mne.io.Raw.filter)``.
# 
# If a function is not a direct MNE method, check the steps osl-ephys registers: ``osl_ephys.pipeline.registry.registered('preproc')`` lists them by their config name, and ``help(osl_ephys.preproc.steps.bad_segments)`` explains the options of one.
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
# :Tip: ``bad_segments`` and ``bad_channels`` call ``osl_ephys.preproc.detect_bad_segments`` and ``detect_bad_channels``; see their help for the options.
#
# :Tip: Fitting ICA and applying it to the data are the steps ``ica_raw``, ``ica_autoreject`` and ``apply_ica``. Check ``help(osl_ephys.preproc.steps.ica_raw)`` and ``help(osl_ephys.preproc.steps.ica_autoreject)`` for the options.
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
# Running a config with run_chain
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# 
# Now we want to apply this preprocessing pipeline to the first file, we do this using the osl-ephys function ``osl_ephys.pipeline.run_chain(config, session, ...)``. A session is a dict with an ``id`` and the files the steps need (here only the ``raw_file``), or just the path of the raw file. Let's first run this function, and then have a more detailed look at its inputs and outputs.



help(run_chain)
session = {"id": "sub001-ses01", "raw_file": sub1run1}
dataset = run_chain(config_text, session, outdir=outdir, overwrite=True)


print(f"run_chain returned a dictionary with the following items: \n {dataset.keys()} \n\n")

#%%
# Let's dive into this function in a bit more detail. 
# There are two required inputs: 
#
# - ``config``: dict, path, or config text - as above
#
# - ``session``: a dict with the session's ``id`` and files (``raw_file``, and later ``smri_file``), a path to a raw file, or a loaded MNE ``Raw``
# 
# The optional inputs are:
#
#
#
# - ``outdir``: The output directory. By default the preprocessed data is not saved. Add a path here if you wish to save it.
#
# - ``log_dir``: The directory for the session's log file (``outdir/logs`` by default).
#
# - ``qc_dir``: The directory for the QC plots and the report (``outdir/qc`` by default). We will have a closer look at the report later.
#
#
#
# - ``overwrite``: Whether or not the overwrite existing data
#
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
# The ``dataset`` dictionary that is returned by ``run_chain`` contains different items depending on the preprocessing pipeline. For example ``dataset["epochs"]`` is only returned because we specified our pipeline to include creating epochs. If we're saving the data to disk (i.e., ``outdir`` is specified), every item in ``dataset`` is saved separately in ``outdir/preproc``. The filenames will look something like:
#
# - ``raw``: `sub001-ses01_preproc-raw.fif`
# - ``events``: `sub001-ses01_events.npy` - numpy's way of saving data (load with `numpy.load()`)
# - ``epochs``: `sub001-ses01_epo.fif`
# - ``event_id``: `sub001-ses01_event-id.yml` - YAML file
# - ``ica``: `sub001-ses01_ica.fif`
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
# and the following input to ``run_chain``:
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

run_chain(config_text, session, outdir=outdir, overwrite=True, extra_funcs=[ica_kurtosisreject])

#%%
# As an alternative to running `run_chain` from within Python, osl-ephys can run a config from the terminal's command line as well, on a CSV file listing the sessions (columns ``id`` and ``raw_file``):
#
# ``osl-ephys run my_config.yml sessions.csv /path/to/my/output_dir --overwrite``
#
#
# Note that this currently is not possible when using custom functions.
# 
# 
# The QC report
# ^^^^^^^^^^^^^
# Every step saves its numbers and plots under ``outdir/qc``, and ``outdir/qc/report.html`` collects them: a table with one row per session (the percentage of data marked bad, the number of bad channels, the ICA components rejected, ...) next to the plots of the selected session. The table can be sorted by any column, which is how to find the sessions worth a closer look when there are many, and values far from the rest of their column are highlighted.
# The idea of the report is to help you optimize your preprocessing pipeline and check the data quality. If a researcher asks you how the quality of the data is, the question is not trivial. With the report, we hope to give you a tool that quantifies some important metrics. It is not exhaustive, so if there's missing anything, please `open an issue on GitHub <https://github.com/OHBA-analysis/osl-ephys/issues>`_. 
# 
# 
# Now open the report. Clicking a row selects a session; the up and down arrows move between sessions and the left and right arrows between the tabs of plots:
# 
# - Preprocessing: the config as a flowchart, the variance over time for each channel type (bad segments shaded red), including and excluding the bad segments and channels, the distribution of the channels' standard deviations (the bad channels marked) and the sensor positions with the bad channels highlighted.
#
# - Spectra: the power spectrum of each channel, per channel type, zoomed in to 1-48 Hz and in full, and topographies of the power in the standard frequency bands.
#
# - Artefacts: the properties of each ICA component labelled as bad (the topography for each sensor type, the power spectrum, the ERP and the variance), the EOG and ECG traces, and the head movement when continuous HPI was recorded.
#
# - Acquisition: the raw data browser view, the digitisation (fiducials, HPI coils, headshape points) and the events.
#
# The session's log, with every step that ran, is in ``outdir/logs/<id>.log``.

#%%
# Optimizing your preprocessing config
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The report is one tool that can help you to optimize your preprocessing, but one other tool you should keep in mind is visualizing the data itself (e.g., using the databrowser ``raw.plot()``).
# In the **Time Series** tab we see for example that there are still points in time with high variance, both in magnetometers and gradiometers. Let's see if we can clean up the data a bit more by adding extra bad segment detection, based on the temporal derivative (i.e., how strong the signal changes; using ``mode: 'diff'```). Let's also try to remove the segment in the EOG where we see a big amplitude increase.  
# 
# Note that here, we drop the custom function for ICA labelling that we used before. We will also set ``apply: false`` in ``ica_autoreject``, so we can manually check the components labeled as bad before we removed them from the data. Therefore, we will also leave out the epoching, since we want to apply this after we're done with ICA. 


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

run_chain(config_text, session, outdir=outdir, overwrite=True)


#%%
# Manually checking ICA
# ^^^^^^^^^^^^^^^^^^^^^
#
# Now that we are happy with the preprocessing pipeline on this dataset, we could load in the data, make manual adjustments to the IC's that were labeled as bad (or at least check them), remove those from the data and save the clean data. We also might want to re-run the report with the updated ICA information.
# In the previous tutorial we used an ICA plotting tool, but it doesn't work in each IDE. We also have a command for this, ``osl-ephys ica review``. You can use this from the terminal (make sure you're in an ``osle`` environment). It takes the output directory of the run and the ids of the sessions to review (every session with an ICA, if none is given), and ``--reject`` says what to do with the components marked as bad when the window is closed: nothing (the default; the labels are saved in the ICA file), ``manual`` (remove only the components labelled in this review; useful if the automatically detected ones were already removed), or ``all`` (remove every labelled component, automatic and manual, from the data). For example:
# 
# ``(osle) > osl-ephys ica review preprocessed sub001-ses01``

#%%
# When we close the figure, the ICA object is saved with the labels, and the ICA plot of the report is updated.
# If we haven't yet removed the components from the data, we can do so afterwards:
#
# ``(osle) > osl-ephys ica apply preprocessed sub001-ses01``
