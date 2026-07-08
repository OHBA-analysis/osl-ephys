"""Use MNE to perform source localisation.

https://mne.tools/stable/auto_tutorials/forward/30_forward.html
"""

import mne

raw_dir = "../data/raw"

subjects_dir = "../data"
subject = "sub-03"

raw_fname = f"{raw_dir}/{subject}/{subject}_ses-meg_task-facerecognition_run-01_proc-sss_meg.fif"
trans_fname = f"{subjects_dir}/{subject}/coreg-trans.fif"
src_fname = f"{subjects_dir}/{subject}/space-src.fif"
fwd_fname = f"{subjects_dir}/{subject}/model-fwd.fif"

# Setup source space (surface)
src = mne.setup_source_space(
    subjects_dir=subjects_dir,
    spacing="oct6",
    subject=subject,
    add_dist="patch",
)
mne.write_source_spaces(src_fname, src, overwrite=True)
print(src)

#plot_bem_kwargs = dict(
#    subject=subject,
#    subjects_dir=subjects_dir,
#    brain_surfaces="white",
#    orientation="coronal",
#    slices=[50, 100, 150, 200],
#)
#fig = mne.viz.plot_bem(src=src, **plot_bem_kwargs)
#fig.savefig("test.png")

#fig = mne.viz.plot_alignment(
#    subject=subject,
#    subjects_dir=subjects_dir,
#    surfaces="white",
#    coord_frame="mri",
#    src=src,
#)
#fig.plotter.export_html("test.html")

# Compute forward model
model = mne.make_bem_model(
    subjects_dir=subjects_dir, subject=subject, ico=5, conductivity=(0.3,)
)
bem = mne.make_bem_solution(model)
trans = mne.read_trans(trans_fname)
fwd = mne.make_forward_solution(
    raw_fname,
    trans=trans,
    src=src,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5.0,
    verbose=True,
)
mne.write_forward_solution(fwd_fname, fwd, overwrite=True)
print(fwd)
