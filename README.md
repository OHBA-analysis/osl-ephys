# OSL: Electrophysiological Data Analysis Toolbox

Tools for analysing electrophysiological (M/EEG) data.

Documentation: https://osl-ephys.readthedocs.io/en/latest/.

## Installation

> **You are looking at the `semp` branch.** On top of upstream osl-ephys it
> bundles two extra subpackages: `osl_ephys.preprocessing.semp` (preprocessing
> for *simultaneous* EEG-fMRI recordings) and `osl_ephys.preprocessing.manual_ica`
> (browser-based manual ICA review). It is in beta — usable, but expect rough
> edges. To install **this** version rather than the released osl-ephys, check
> out the `semp` branch and install from it; the commands below already do that
> (`git clone -b semp ...`). The env files pull in one extra runtime dependency,
> [`osl-pathfinder`](https://test.pypi.org/project/osl-pathfinder/) — which semp
> pipelines use to map recording ids to file paths (see the EEG-fMRI tutorial).
> Note `osl-pathfinder` is published on **TestPyPI**, not PyPI; the env files add
> `--extra-index-url https://test.pypi.org/simple/` so `mamba env create` fetches
> it automatically. To install it by hand:
> `pip install --extra-index-url https://test.pypi.org/simple/ osl-pathfinder`.

We recommend installing osl-ephys in a conda environment.

### Conda / mamba

Miniforge (`conda`/`mamba`) can be installed with:
```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
rm Miniforge3-$(uname)-$(uname -m).sh
```

### osl-ephys

osl-ephys can be installed from source code in a conda environment using the following.

```
git clone -b semp https://github.com/OHBA-analysis/osl-ephys.git
cd osl-ephys
mamba env create -f envs/osle.yml
conda activate osle
pip install -e .
```

Note, on a headless server you may need to set the following environment variable:
```
export PYVISTA_OFF_SCREEN=true
```

### Oxford-specific computers

If you are installing on an OHBA workstation computer (hbaws) use:
```
git clone -b semp https://github.com/OHBA-analysis/osl-ephys.git
cd osl-ephys
mamba env create -f envs/hbaws.yml
conda activate osle
pip install -e .
```

Or on the BMRC cluster:
```
git clone -b semp https://github.com/OHBA-analysis/osl-ephys.git
cd osl-ephys
mamba env create -f envs/bmrc.yml
conda activate osle
pip install -e .
```

Remember to set the following environment variable:
```
export PYVISTA_OFF_SCREEN=true
```

## Removing osl-ephys

Simply remove the conda environment and delete the repository:
```
conda env remove -n osle
rm -rf osl-ephys
```

## For developers

Install all the requirements:
```
pip install -r requirements.txt
```

Run tests:
```
cd osl_ephys
pytest tests
```
or to run a specific test:
```
cd osl_ephys/tests
pytest test_file_handling.py
```

Build documentation locally:
```
sphinx-build -b html doc/source build
```
Compiled docs can be found in `doc/build/html/index.html`.
