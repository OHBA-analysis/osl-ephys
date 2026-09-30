Installation
============

A full installation of the osl-ephys toolbox includes:

- `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_ (FMRIB Software Library) - only needed if you want to do volumetric source reconstruction.
- `FreeSurfer <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_ (FreeSurfer) - only needed if you want to do surface-based source reconstruction.
- `Miniforge <https://conda-forge.org/download/>`_.
- `osl-ephys <https://github.com/OHBA-analysis/osl-ephys>`_ (OSL Ephys Toolbox).

Instructions
------------

1. Install FSL using the instructions `here <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_.

If you're using a Windows machine, you will need to install the above in `Ubuntu <https://ubuntu.com/wsl>`_ using a Windows subsystem. Make sure to setup XLaunch for visualisations.

2. Install Freesurfer using the instructions `here <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_.

3. Install Miniforge3 with::

    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    rm Miniforge3-$(uname)-$(uname -m).sh

Note, if you're using a Windows computer, you will need to do this in the WSL Ubuntu terminal that was used to install FSL (step 1).

4. Clone OSL-Ephys, create its environment, and install the checkout::

    git clone https://github.com/OHBA-analysis/osl-ephys.git
    cd osl-ephys
    mamba env create -f envs/osle.yml
    conda activate osle
    python -m pip install -e .

This creates an environment called :code:`osle` and installs the code from
this checkout, including SEMP. For OHBA workstations or BMRC, use the
corresponding environment file and activation name in the repository README.

Loading the packages
--------------------

To use osl-ephys you need to activate the conda environment::

    conda activate osle

**You need to do every time you open a new terminal.** You know if the :code:`osle` environment is activated if it says :code:`(osle)[...]` at the start of your terminal command line.

Note, if you get a :code:`conda init` error when activating the :code:`osle` environment during a job on an HPC cluster, you can resolve this by replacing::

    conda activate osle

with::

    source activate osle

Integrated Development Environments (IDEs)
------------------------------------------

The osl-ephys installation comes with `Jupyter Notebook <https://jupyter.org/>`_. To open Jupyter Notebook use::

    conda activate osle
    jupyter notebook

Test the installation
---------------------

The following should not raise any errors::

    conda activate osle
    python
    >> import osl_ephys

Update the source checkout
--------------------------

The editable installation in step 4 uses your local checkout. To update it
after changes have been merged into the repository::

    cd osl-ephys
    git pull
    conda activate osle
    python -m pip install -e .

Getting help
------------

If you run into problems while installing osl-ephys, please open an issue on the `GitHub repository <https://github.com/OHBA-analysis/osl-ephys/issues>`_.
