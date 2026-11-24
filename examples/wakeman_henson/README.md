# Wakeman-Henson face recognition dataset

The analysis of the osl-ephys toolbox paper (van Es et al., 2025,
https://doi.org/10.3389/fnins.2025.1522675) on the 19 subjects and 6 runs
of the face recognition dataset of Wakeman and Henson (OpenfMRI ds000117,
https://openfmri.org/s3-browser/?prefix=ds000117/ds000117_R0.1.1/compressed/).
Extract the archives into a directory called `ds117` next to the scripts,
and install FSL. The scripts run in the order they appear in the paper:

1. `1_preprocess.py`: events, filtering, downsampling, bad segments and
   channels, and an ICA whose components are labelled by the EOG and ECG
   correlation but not yet removed. In the paper the labels of eight
   sessions were refined by hand (`osl-ephys ica review data <id> --reject
   all`, for the ids listed in the script); every other session is then
   cleaned with `osl-ephys ica apply data`.
2. `2_coregister.py`: surfaces and coregistration.
3. `3_source_reconstruct.py`: the forward model, beamforming and
   parcellation with the 52-parcel Glasser atlas.
4. `4_sign_flip.py`: the dipole sign ambiguity across sessions.
5. `5_stats.py`: a GLM per session on the epoched parcel time courses
   (famous, unfamiliar and scrambled faces) and a group GLM with a
   max-statistic permutation test of the mean response.

`2_coregister_freesurfer.py` and `3_source_reconstruct_freesurfer.py`
are the surface-based alternative to steps 2 and 3: FreeSurfer's
recon-all and watershed BEM, MNE's coregistration, then a surface forward
model, minimum norm and the aparc parcellation. They need FreeSurfer, and
recon-all takes hours per subject.

To reproduce the paper exactly, set `random_seed` in `run_batch` to the
seeds in the log files on OSF (https://osf.io/2rnyg/); the environment the
paper used is in the git history of osl-ephys 2.4 (`examples/toolbox-paper`).
