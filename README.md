# OSL: Electrophysiological Data Analysis Toolbox

Tools for analysing electrophysiological (M/EEG) data.

This `semp` branch adds simultaneous EEG-fMRI preprocessing and browser-based
manual ICA review to OSL-Ephys. It currently targets Python 3.9 or newer and
`osl-pathfinder==1.0.0`.

## Install the SEMP branch

Using [Miniforge](https://github.com/conda-forge/miniforge) with `mamba` is
recommended. First clone the SEMP branch:

```bash
git clone --branch semp https://github.com/OHBA-analysis/osl-ephys.git
cd osl-ephys
```

Choose the environment file for the computer you are using.

### OHBA workstations (HBAWS)

```bash
mamba env create -f envs/hbaws.yml
conda activate semp
python -m pip install -e .
```

### BMRC

```bash
mamba env create -f envs/bmrc.yml
conda activate osle
python -m pip install -e .
```

### Other Linux/MacOS computers

```bash
mamba env create -f envs/osle.yml
conda activate osle
python -m pip install -e .
```

Confirm the active installation with:

```bash
python -c "import osl_ephys, osl_pathfinder; print(osl_ephys.__version__, osl_pathfinder.__version__)"
```

On a headless server, set:

```bash
export PYVISTA_OFF_SCREEN=true
```

## SEMP tutorials

- [Preprocessing simultaneous EEG-fMRI](doc/source/tutorials/preprocessing_eeg-fmri.py): download a small NATVIEW example, inspect its acquisition metadata, run each SEMP stage, validate sensor-space spectra, then express the SEMP workflow as a batch config.
- [Manual ICA review](doc/source/tutorials/preprocessing_manual-ica.py): fit ICA in a batch, review components in the browser, and apply the recorded decisions safely.

To regenerate their Jupyter notebooks from the tutorial Python files:

```bash
python -m pip install -e ".[doc]"
sphinx-build -b html -D plot_gallery=0 doc/source build/html
ls doc/source/tutorials_build/preprocessing_{eeg-fmri,manual-ica}.ipynb
```

Sphinx-Gallery writes the two notebooks under `doc/source/tutorials_build/`.
`plot_gallery=0` converts the files without executing the data-dependent
tutorial code.

The full project documentation is at
[osl-ephys.readthedocs.io](https://osl-ephys.readthedocs.io/en/latest/).

## Develop

```bash
python -m pip install -e ".[full]"
pytest osl_ephys/tests
sphinx-build -b html -D plot_gallery=0 doc/source build/html
```

The final command validates documentation structure without executing tutorials
that require user-supplied datasets. Generated HTML is written to
`build/html/index.html`.
