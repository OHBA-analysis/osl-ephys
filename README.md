# OHBA Software Library: Electrophysiology Toolbox

[![PyPI version](https://img.shields.io/pypi/v/osl-ephys)](https://pypi.org/project/osl-ephys/)
[![Documentation](https://readthedocs.org/projects/osl-ephys/badge/?version=latest)](https://osl-ephys.readthedocs.io)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](https://github.com/OHBA-analysis/osl-ephys/blob/main/LICENSE)
[![Paper](https://img.shields.io/badge/paper-Frontiers-orange)](https://doi.org/10.3389/fnins.2025.1522675)

osl-ephys is a Python toolbox for analysing electrophysiological data (MEG, EEG and OPM) from the raw recording to parcellated source-space time courses. It builds on [MNE-Python](https://mne.tools) and adds the pieces an M/EEG pipeline needs around it:

- **Preprocessing** driven by a YAML config: filtering, bad segment and bad channel detection, ICA artefact rejection, epoching, with the same config run on one session or batched over a dataset in parallel.
- **MaxFilter** wrapper for Elekta/MEGIN data (SSS and tSSS), with multistage and chunked runs.
- **Coregistration and source reconstruction** with RHINO: surface extraction from a structural MRI with FSL, coregistration to the head shape, forward modelling, LCMV beamforming and volumetric minimum norm.
- **Parcellation** of source-space data into parcel time courses, with spatial leakage correction and sign flipping across sessions.
- **Quality control**: an HTML report across sessions with the plots and numbers of every step.
- **Sensor-level statistics**: GLM-Spectrum and GLM-Epochs with permutation testing.

For a full description of the toolbox, see the [documentation](https://osl-ephys.readthedocs.io).

## Example

One config names the steps of each stage, and `run_batch` runs it on a set of sessions in parallel, writing a log per session and a QC report:

```python
from osl_ephys.pipeline import run_batch

config = """
preproc:
  - filter: {l_freq: 1, h_freq: 45}
  - resample: {sfreq: 250}
  - bad_segments: {picks: mag}
  - bad_channels: {picks: mag}
  - ica_label: {picks: mag}
coreg:
  - extract_surfaces: {include_nose: false}
  - extract_fiducials_and_headshape: {}
  - coregister: {use_nose: false}
source_recon:
  - forward_model: {model: Single Layer}
  - beamform: {chantypes: mag, rank: {mag: 60}}
parcellation:
  - parcellate: {parcellation_file: atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz}
group:
  - sign_flip: {}
"""

sessions = [
    {"id": "sub-01", "subject": "sub-01", "raw_file": "raw/sub-01.fif", "smri_file": "smri/sub-01.nii.gz"},
    {"id": "sub-02", "subject": "sub-02", "raw_file": "raw/sub-02.fif", "smri_file": "smri/sub-02.nii.gz"},
]

run_batch(config, sessions, "derivatives", n_workers=2)
```

The same run from the terminal, with the sessions in a CSV file: `osl-ephys run config.yml sessions.csv derivatives --n-workers 2`. The parcel time courses end up in `derivatives/<id>/`, the report in `derivatives/qc/report.html`.

## Installation

osl-ephys is on PyPI and needs Python 3.10 or newer. Install it into its own environment, e.g.
```
conda create -n osle python=3.12
conda activate osle
pip install osl-ephys
```

See the [installation page](https://osl-ephys.readthedocs.io/en/latest/install.html) for more information, including headless servers.

## Documentation

The read the docs page is updated automatically whenever there is a new commit on the `main` branch.

The documentation is included as docstrings in the source code, and the `doc/` directory contains `.rst` files for installation, tutorials and the FAQ. To build the documentation locally:
```
cd osl-ephys
pip install -r doc/requirements.txt
sphinx-build -b html doc build
```
The local build can be found in `build/index.html`.

## Releases

To release a new version:

1. Check the latest commit on `main` has compiled successfully on [readthedocs](https://readthedocs.org/projects/osl-ephys).

2. Create a new release using the 'Create a new release' link on the right of the GitHub repo webpage:

    - Set the tag to the new version number with a `v` prefix (e.g. `v3.0.0`).
    - Write the release notes.
    - Select 'Latest' for the release label and click 'Publish release'.

3. Publishing the release triggers the `release.yml` GitHub Actions workflow, which builds the package and uploads it to [PyPI](https://pypi.org/project/osl-ephys/). Check it succeeded under the Actions tab of the GitHub repo.

## Citation

If you find this toolbox useful, please cite the [paper](https://doi.org/10.3389/fnins.2025.1522675):

> **van Es, M. W. J., Gohil, C., Quinn, A. J., & Woolrich, M. W. (2025). osl-ephys: a Python toolbox for the analysis of electrophysiology data. Frontiers in Neuroscience, 19, 1522675.**

osl-ephys builds on MNE-Python, and the source reconstruction on FSL. Please cite those packages as well when you use them.
