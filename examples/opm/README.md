# Nottingham OPM Example

Preprocessing and source reconstruction of OPM data.

Data can be downloaded from: https://zenodo.org/doi/10.5281/zenodo.7525341. Note you need to unzip the files using 7zip if you have a Mac computer.

## Pipeline

- `1_preprocess.py`: Preprocesses the data. This includes filtering, downsampling and automated artefact removal.
- `2_coregister.py`: Extract surfaces from the structural MRI, create the coregistration files OSL is expecting.
- `3_source_reconstruct.py`: Calculate forward model, beamform and parcellate.
- `4_sign_flip.py` (optional): Sign flipping to fix the dipole sign ambiguity.
