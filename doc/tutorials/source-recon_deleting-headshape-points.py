"""
Deleting Headshape Points
=========================

One cause of bad co-registrations can be due to the presence of misleading or erroneous headshape points. These can be caused by errors in the recording of the headshape points when the experimenter was using the polhemus system.

In this tutorial we use :func:`~osl_ephys.rhino.delete_headshape_points`, which shows an interactive figure of the polhemus-derived headshape points in polhemus space: points can be clicked on to delete them, and the changes are written to the session's headshape file.

Typically, you would use this after running the coregistration (either directly or via the batch API), and after diagnosing a bad coreg (from the coregistration plot or the QC report).

To put ourselves in this situation we will first download the appropriate data and copy the headshape points to the paths RHINO reads them from.

Download files
**************

:note: To download the dataset you need ``osfclient`` installed: ``pip install osfclient``.
"""

import os
import os.path as op

import numpy as np

from osl_ephys import io, rhino
from osl_ephys.session import Session


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
# Setup file names
# ****************

fsl_dir = '~/fsl'
rhino.setup_fsl(fsl_dir)

data_dir = './notts_2subjects'
outdir = './notts_2subjects/recon'

subject = '{subject}'
fif_files = io.Study(op.join(data_dir, subject, subject + '_task-resteyesopen_meg_preproc_raw.fif'))
subjects = fif_files.fields['subject']
fif_files = fif_files.get()

session = Session(
    outdir=outdir,
    id=subjects[0],
    surfaces_dir=op.join(outdir, "surfaces", subjects[0]),
    preproc_file=fif_files[0],
)

#%%
# Copy the polhemus files to the session's coregistration directory
# ******************************************************************

os.makedirs(session.coreg_dir, exist_ok=True)
polhemus_dir = op.join(data_dir, session.id, 'polhemus')
for name in ['nasion', 'rpa', 'lpa', 'headshape']:
    points = np.loadtxt(op.join(polhemus_dir, f'polhemus_{name}.txt'))
    np.savetxt(getattr(session.coreg, f'head_{name}_file'), points)

print('E.g., the coordinates for the nasion for subject {} in Polhemus space are \n'.format(subjects[0]))
print(np.loadtxt(session.coreg.head_nasion_file))

#%%
# Delete headshape points
# ***********************
# 
# We can now call :func:`~osl_ephys.rhino.delete_headshape_points`, with the session (it reads and writes ``session.coreg.head_headshape_file``) or with the path of any headshape file (``headshape_file=...``, a text file holding the (3 x num_headshapepoints) array of points).

rhino.delete_headshape_points(session)

#%%
# This brings up an interactive figure of the polhemus derived headshape points in polhemus space, as a scatter plot.
# 
# - The plot can be rotated (avoid clicking near an actual headshape point when doing this).
# - Headshape points can be clicked on to delete them.
# - The figure should be closed upon completion, at which point there is the option to save the changes.
# 
# Go ahead and delete some headshape points, after which close the figure, and then choose to save the file.
# Since we have worked on the headshape points file of the session, the saved file will be the one used by any subsequent coregistration, e.g., when we call:

rhino.coregister_head_and_mri(session, use_headshape=True, use_nose=True)
