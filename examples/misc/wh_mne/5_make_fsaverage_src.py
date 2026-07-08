"""Create fsaverage source space.

"""

import os
import mne

subjects_dir = "../data"
subject = "fsaverage"

os.makedirs(f"{subjects_dir}/morph-maps", exist_ok=True)

src_fname = f"{subjects_dir}/morph-maps/fsaverage-src.fif"

src = mne.setup_source_space(
    subjects_dir=subjects_dir,
    subject=subject,
    spacing="oct6",
    add_dist="patch",
)
mne.write_source_spaces(src_fname, src, overwrite=True)
