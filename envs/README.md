# Conda Environments

- `osle.yml`: for Linux or MacOS computers.
- `hbaws.yml`: for Oxford OHBA workstation computers.
- `bmrc.yml`: for the Oxford BMRC cluster.

For a general environment, run:
```
git clone https://github.com/OHBA-analysis/osl-ephys.git
cd osl-ephys
conda env create -f envs/osle.yml
conda activate osle
pip install -e .
```

On HBAWS, substitute `envs/hbaws.yml`; on BMRC, substitute `envs/bmrc.yml`.
All three files create an environment named `osle`.

All environments come with Jupyter Notebook.
