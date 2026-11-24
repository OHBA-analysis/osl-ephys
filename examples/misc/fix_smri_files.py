"""Give structural MRIs a standard sform.

Only needed when RHINO raises an error about the sform code of an MRI. The
fixed copies go to smri/fixed and the sessions should point at those.
"""

import os
from pathlib import Path
from shutil import copyfile

import nibabel as nib
import numpy as np

from osl_ephys.utils.misc import system_call

smri_files = ["smri/sub-001_T1w.nii.gz", "smri/sub-002_T1w.nii.gz"]
fixed_dir = "smri/fixed"
os.makedirs(fixed_dir, exist_ok=True)

for smri_file in smri_files:
    fixed_file = f"{fixed_dir}/{Path(smri_file).name}"
    copyfile(smri_file, fixed_file)
    sform = np.copy(nib.load(fixed_file).header.get_sform())
    sform[0, :4] = [-1, 0, 0, 128]
    sform[1, :4] = [0, 1, 0, -128]
    sform[2, :4] = [0, 0, 1, -90]
    system_call(
        f"fslorient -setsform {' '.join(map(str, sform.flatten()))} {fixed_file}"
    )
    print("Fixed:", fixed_file)
