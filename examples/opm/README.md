# OPM example (Nottingham)

Preprocessing and source reconstruction of the Nottingham OPM dataset,
https://zenodo.org/doi/10.5281/zenodo.7525341 (unzip with 7zip on a Mac).

1. `0_convert_files.py`: the MATLAB files to fif, with the sensor positions
   and the head-to-MRI transform the recording comes with.
2. `1_preprocess.py`: filtering, downsampling, interference rejection with
   the adaptive multipole model and artefact removal.
3. `2_coregister.py`: the surfaces from the MRI; the recording is already
   coregistered, so the step only writes the files the forward model needs.
4. `3_source_reconstruct.py`: the forward model, beamforming and
   parcellation.
5. `4_sign_flip.py`: the dipole sign ambiguity across sessions.
