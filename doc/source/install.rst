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

4. Install osl-ephys::

    curl https://raw.githubusercontent.com/OHBA-analysis/osl-ephys/main/envs/osle.yml > osle.yml
    mamba env create -f osle.yml
    rm osle.yml

This will create a conda environment called :code:`osle`.

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

    conda activate osl
    jupyter notebook

Test the installation
---------------------

The following should not raise any errors::

    conda activate osle
    python
    >> import osl_ephys

Get the latest source code (optional)
-------------------------------------

If you want the very latest code you can clone the GitHub repo. This is only neccessary if you want recent changes to the package that haven't been released yet.

First install osl-ephys using the instructions above. Then clone the repo and install locally from source::

    conda activate osle

    git clone https://github.com/OHBA-analysis/osl-ephys.git
    cd osl-ephys
    pip install -e .
    cd ..

After you install from source, you can run the code with local changes. You can update the source code using::

    git pull

within the :code:`osl-ephys` directory.

Getting help
------------

If you run into problems while installing osl-ephys, please open an issue on the `GitHub repository <https://github.com/OHBA-analysis/osl-ephys/issues>`_.
