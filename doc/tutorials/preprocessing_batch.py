"""
Batch preprocessing and parallelization
=======================================

In this tutorial we will use the config that we created in the previous tutorial for batch processing all data in one call (it should be located in `Preprocessing/config.yaml` - if not, you can download it `here <https://osf.io/vgxub>`_). We will also see how the computations are spread over the cores of your computer.

This tutorial looks as follows:

1. **Batch processing**
2. **The report across sessions**
3. **Parallelization**
4. **Concluding remarks**
"""

#%%
# Batch processing
# ^^^^^^^^^^^^^^^^
# Once we have a preprocessing config that we think is effectively cleaning our data, we will use it on more datasets. Initially, we might want to still limit it to a few datasets, but when we're confident we will want to preprocess all data.
# Rather than running ``run_chain`` for every session separately, we can achieve batch processing with a single call to ``run_batch`` (which runs ``run_chain`` on every session and does some bookkeeping). We will first run ``run_batch`` on run 1 of subjects 1-3. Note that this function never returns the ``dataset`` (this would be too much data to keep in memory). Instead, the data is saved to disk, and the function returns the status of every session (``ok``, ``partial`` when a step warned, or ``failed``).

import os
from pprint import pprint

import osl_ephys
from osl_ephys.pipeline import run_batch
from osl_ephys.pipeline.config import load_config

basedir = os.getcwd()
outdir = os.path.join(basedir, "preprocessed")

# There are multiple runs for each subject. We will first fetch all data using an osl-ephys utility
name = '{subj}_ses-meg_task-facerecognition_{run}_meg'
fullpath = os.path.join(basedir, 'data', name + '.fif')
datafiles = osl_ephys.io.Study(fullpath)

# The sessions: run 1 of subjects 1-3, each a dict with an id and its raw file
sessions = [
    {"id": f"sub{i:03d}-run01", "raw_file": datafiles.get(subj=f'sub-0{i}')[0]}
    for i in range(1, 4)
]
print('Found {} sessions:'.format(len(sessions)))
pprint(sessions)

#%%
# We'll load in the config that we created in the previous tutorial, and change the ICA settings to speed up processing.

config = load_config(os.path.join(basedir, "config.yaml"))
config['preproc'][-2]['ica_raw']['n_components'] = 20
config

# process subjects 1-3
results = run_batch(config, sessions, outdir, overwrite=True)
pprint(results)

#%% 
# The report across sessions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# In the previous tutorial we have already seen the QC report. Now that we've processed multiple sessions, its table has one row per session, with the metrics of each:
#
# - the percentage of the recording marked as a bad segment, overall and per channel type;
# - the number of bad channels;
# - the number of ICA components rejected;
# - the number of EOG and ECG events found, and the head movement when continuous HPI was recorded.
#
# Sorting the table by a metric brings the anomalous sessions to the top, and the "highlighted only" box keeps the rows with a value far from the rest. This is how to direct your attention to specific data when processing large amounts of it. Clicking a row shows that session's plots, the same tabs as before. In this particular case we see that many sessions have quite a few bad channels removed.
#
# Next to the report, ``outdir/logs`` has a log per session and ``summary.csv`` with the status and run time of each.
#
# Based on the report on all runs from subject 1 it still seems we have a reasonable preprocessing pipeline. If your dataset is particularly large, you might now want to run the same config on a couple more subjects before you process the entire dataset. Here, we will process run 1 from all 16 subjects. 
#
#
# 
# Parallelization
# ^^^^^^^^^^^^^^^
# ``run_batch`` processes ``n_workers`` sessions at the same time, each in its own process. As a rule of thumb, use half the cores of your computer; if you're working on a shared computing system, discuss the appropriate setting with your IT support.
#
# Because the sessions run in separate processes, the script has to be guarded by ``if __name__ == '__main__':``, so that the worker processes do not run it again when they import it:

if __name__ == '__main__':
    import os
    import osl_ephys
    from osl_ephys.pipeline import run_batch

    n_cores = 4  # we assume you at least have 4 cores, we'll use half of these

    basedir = os.getcwd()
    outdir = os.path.join(basedir, "preprocessed")
    name = '{subj}_ses-meg_task-facerecognition_{run}_meg'
    datafiles = osl_ephys.io.Study(os.path.join(basedir, 'data', name + '.fif'))
    sessions = [
        {"id": f"sub{i + 1:03d}-run01", "raw_file": datafiles.get(subj=f'sub-{i + 1:02d}', run='run-01')[0]}
        for i in range(16)
    ]
    config = os.path.join(basedir, "config.yaml")
    run_batch(config, sessions, outdir, n_workers=n_cores // 2)

#%%
# The same run from the terminal, with the sessions in a CSV file (columns ``id`` and ``raw_file``):
#
# ``(osle) > osl-ephys run config.yaml sessions.csv preprocessed --n-workers 2``

#%% 
# Now open the report and see how well the preprocessing went.
# The table shows that we're throwing away a lot of data. Sessions have up to 20% of bad segments and some have more than 10 bad channels. This indicates that either the preprocessing pipeline is too rigorous, or the data is very noisy (Note that these options are ill-defined and depend on your criteria!)
#
# Concluding remarks
# ^^^^^^^^^^^^^^^^^^
# You have now learned how to preprocess your MEG/EEG data, and that it typically requires some iterations before you find a preprocessing pipeline that works well for your data. This is entirely dependent on your criteria for how clean you need the data to be, taking into account how much data you want to retain, and how much time you have to optimize the pipeline.
# We've also seen that osl-ephys offers functionality on top of MNE-Python. Some of them are the unique and concise ``config`` structure, additional (preprocessing) functions, the QC report, and parallel batch processing. Because both mainly work with ``.fif`` files, they are very well integrated, as you will see in the following tutorials.
