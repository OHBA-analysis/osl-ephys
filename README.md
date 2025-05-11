# OSL: Electrophysiological Data Analysis Toolbox

Tools for analysing electrophysiological (M/EEG) data.

Documentation: https://osl-ephys.readthedocs.io/en/latest/.

## Installation

We recommend installing osl-ephys in a conda environment.

### Conda / mamba

Miniforge (`conda`) can be installed with:
```
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
rm Miniforge3-$(uname)-$(uname -m).sh
```

Mamba (`mamba`) can be installed with:
```
conda install -n base -c conda-forge mamba
```

### osl-ephys

osl-ephys can be installed from source code in a conda environment using the following.

```
git clone https://github.com/OHBA-analysis/osl-ephys.git
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
git clone https://github.com/OHBA-analysis/osl-ephys.git
cd osl-ephys
mamba env create -f envs/hbaws.yml
conda activate osle
pip install -e .
```

Or on the BMRC cluster:
```
git clone https://github.com/OHBA-analysis/osl-ephys.git
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
