# EEG example

Source reconstruction of EEG data with a structural MRI but no digitised
electrode positions: the electrodes of the montage are used as the
headshape.

1. `1_preprocess.py`: filtering, downsampling and artefact removal.
2. `2_coregister.py`: surfaces and coregistration. Check in the
   coregistration tab of the report that the electrodes sit on the scalp.
3. `3_source_reconstruct.py`: the three-layer forward model, beamforming
   and parcellation.
4. `4_sign_flip.py`: the dipole sign ambiguity across sessions.
