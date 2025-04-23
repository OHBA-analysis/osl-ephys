'''
Batch preprocessing and parallelization
=======================================

In this tutorial we will use the config that we created in the previous tutorial for batch processing all data in one call (it should be located in `Preprocessing/config.yaml` - if not, you can download it `here <https://osf.io/vgxub>`_. We will also explore how you could use multiple cores of your computer to parallelize the computations.
This tutorial looks as follows:

1. **Batch processing**
2. **The Summary report**
3. **Parallelization using Dask**
4. **Concluding remarks**

'''

#%%
# Batch processing
# ^^^^^^^^^^^^^^^^
# Once we have a preprocessing config that we think is effectively cleaning our data, we will use it on more datasets. Initially, we might want to still limit it to a few datasets, but when we're confident we will want to preprocess all data.
# Rather than running ``osl_proc_chain`` for every dataset seperately, we can achieve batch processing with a single call to ``osl_proc_batch`` (which internally loops over ``run_proc_chain`` and does some bookkeeping). We will first run ``osl_proc_batch`` on all run 1 from subjects 1-3. Note that this function never returns the ``dataset` (this would be too much data to keep in memory). Instead, the data is saved to disk, and the function returns ``True`` for every successfully processed dataset, and ``False`` for every failed dataset.


import osl_ephys
from osl_ephys.preprocessing import run_proc_batch
import numpy as np
import matplotlib.pyplot as plt
import os
from pprint import pprint

basedir = os.getcwd()
outdir = os.path.join(basedir, "preprocessed")
# generate the output directory if it doesn't exist
os.makedirs(outdir, exist_ok=True)

# There are multiple runs for each subject. We will first fetch all data using an osl-ephys utility
name = '{subj}_ses-meg_task-facerecognition_{run}_meg'
fullpath = os.path.join(basedir, 'data', name + '.fif')
datafiles = osl_ephys.utils.Study(fullpath)

# Find the files to subjects 1-3
infiles = [datafiles.get(subj=f'sub-0{i}')[0] for i in range(1,4)]

subjects_ids = [f"sub{i+1:03d}-run{j+1:02d}" for i in range(3) for j in range(1)]
print('Found {} files:'.format(len(infiles)))
pprint(infiles)

#%%
# We'll load in the config that we created in the previous tutorial, and change the ICA settings to speed up processing.


config = osl_ephys.preprocessing.load_config(os.path.join(basedir, "config.yaml"))# load in the config
config['preproc'][-4]['ica_raw']['n_components'] = 20
config

# process subjects 1-3
run_proc_batch(config, infiles, subjects=subjects_ids, outdir=outdir, overwrite=True) 


#%% 
# The Summary Report
# ^^^^^^^^^^^^^^^^^^
# In the previous tutorial we have already seen the HTML subject report. Now that we've processed multiple datasets, we can browse through each dataset's report. This will show the following.
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
#
#
# But ``run_proc_batch`` Also generates a summary report. This report contains summary metrics that can guide us into finding anomalies. This is especially helpful when we're processing large amounts of data. You can use the preproc summary table to sort by e.g. bad segments to find a particular dataset with a lot of bad segments, so you know you have to further investigate that particular dataset (i.e., by checking that dataset's subject report). In this particular case we see that many datasets have quite a few bad channels removed.
# 
# - Config: The configuration used, as well as any manually defined functions that were supplied to osl-ephys
#
# - Preproc Summary: A quantitative summary table that is interactive, and can be used to guide quality assurance (QA), e.g., do direct your attention to specific data.
#
# - Batch Log: the log file of the batch processing
#
# - Error Logs (optional): If any errors occurred and processing of one of the data files failed, you can find the error files here.
#
# Based on the report on all runs from subject 1 it still seems we have a reasonable preprocessing pipeline. If your dataset is particularly large, you might now want to run the same config on a couple more subjects before you process the entire dataset. Here, we will process run 1 from all 16 subjects. 
#
#
# 
# Parallelization using Dask
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# By default, ``run_proc_batch`` loops the preprocessing over datasets. But most computers nowadays have multiple cores, which means we can use different cores to process different datasets simultaneously. We are using `Dask <https://www.dask.org/>`_ for scheduling and bookkeeping these tasks (we don't want different cores to process the same file!).
# We recommend that you only use Dask after having discussed this with your local IT support. This is because Dask has a lot of power if you give it the wrong settings and it could jam your computer. This is even more important if you're working on a shared computing system. 
# 
# There are some extra things we need to set up in order to use Dask. Firstly, we need to put our main script in a function that looks as follows:
# 

if __name__ == '__main__':
    from dask.distributed import Client
    client = Client(threads_per_worker=1, n_workers=4)
    
    # write extra code here, e.g., definitions of config, files, output_dir
    
    osl_ephys.preprocessing.run_proc_batch(config, 
                                     infiles, 
                                     subjects=subjects_ids, 
                                     outdir=outdir, 
                                     dask_client=True)


# %%
# 
# We will now process the first run of all 16 subjects using Dask. The ``Client`` is how we interact with the cores that we can use. 
# 
# :note: ``threads_per_worker`` should always be set to 1. ``n_workers`` depends on your computing infrastructure. For example, if you're on a personal computer with 8 cores, you can at most use ``n_workers=8``. If you're working on a shared computing infrastructure, discuss the appropriate setting with your IT support. As a rule of thumb, here we will use half the cores that are available on your computer.
# 


if __name__ == '__main__':
    from dask.distributed import Client
    import glob
    import osl_ephys
    import numpy as np
    import os

    n_cores = 4 # we assume you at least have 4 cores, we'll use half of these
    
    client = Client(threads_per_worker=1, n_workers=int(n_cores/2))
    
    # write extra information here, e.g., definitions of config, files, output_dir
    basedir = os.getcwd()
    outdir = os.path.join(basedir, "preprocessed")
    os.makedirs(outdir, exist_ok=True)
    
    name = '{subj}_ses-meg_task-facerecognition_{run}_meg'
    fullpath = os.path.join(basedir, 'data', name + '.fif')
    datafiles = osl_ephys.utils.Study(fullpath)
    
    filenames = datafiles.get()
    subjects_ids = [f"sub{i+1:03d}-run{j+1:02d}" for i in range(16) for j in range(1)]
    config = os.path.join(basedir, "config.yaml")
    
    osl_ephys.preprocessing.run_proc_batch(config, filenames, subjects=subjects_ids, outdir=outdir, dask_client=True)

#%% in order to run this code, we can put it in a script and run it from the terminal.

import os
with open("run_batch_preprocessing.py", "w") as f:
    f.write("""\
if __name__ == '__main__':
    from dask.distributed import Client
    import glob
    import osl_ephys
    import numpy as np
    import os

    n_cores = 4 # we assume you at least have 4 cores, we'll use half of these
    
    client = Client(threads_per_worker=1, n_workers=int(n_cores/2))
    
    # write extra information here, e.g., definitions of config, files, output_dir
    basedir = os.getcwd()
    outdir = os.path.join(basedir, "preprocessed")
    os.makedirs(outdir, exist_ok=True)
    
    name = '{subj}_ses-meg_task-facerecognition_{run}_meg'
    fullpath = os.path.join(basedir, 'data', name + '.fif')
    datafiles = osl_ephys.utils.Study(fullpath)
    
    filenames = datafiles.get()
    subjects_ids = [f"sub{i+1:03d}-run{j+1:02d}" for i in range(16) for j in range(1)]
    config = os.path.join(basedir, "config.yaml")
    
    osl_ephys.preprocessing.run_proc_batch(config, filenames, subjects=subjects_ids, outdir=outdir, dask_client=True)

    """)
    
# Run the script from the terminal
import subprocess
proc = subprocess.Popen('python run_batch_preprocessing.py', stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
output, _ = proc.communicate()



#%% 
# Now open the summary and subject report and see how well the preprocessing went.
# The summary shows that we're throwing away a lot of data. Datasets have up to 20% of badsegments and some have more than 10 bad channels. This indicates that either the preprocessing pipeline is too rigorous, or the data is very noisy (Note that these options are ill-defined and depend on your criteria!)
#
# Concluding remarks
# ^^^^^^^^^^^^^^^^^^
# You have now learned how to preprocess your MEG/EEG data, and that it typically requires some iterations before you find a preprocessing pipeline that works well for your data. This is entirely dependent on your criteria for how clean you need the data to be, taking into account how much data you want to retain, and how much time you have to optimize the pipeline.
# We've also seen that osl-ephys offers functionality on top of MNE-Python. Some of them are the unique and concise ``config`` structure, additional (preprocessing) functions, the preprocessing report, and the option for (parallel) batch processing. Because both mainly work with ``.fif`` files, they are very well integrated, as you will see in the following tutorials. 
