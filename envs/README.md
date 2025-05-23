# Conda Environments

- `osle.yml`: for Linux or MacOS computers.
- `hbaws.yml`: for Oxford OHBA workstation computers.
- `bmrc.yml`: for the Oxford BMRC cluster.

These can be install with:
```
git clone https://github.com/OHBA-analysis/osl-ephys.git
cd osl-ephys
conda env create -f envs/<os>.yml
conda activate osle
pip install -e .
```

All environments come with Jupyter Notebook.
