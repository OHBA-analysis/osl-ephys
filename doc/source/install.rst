Installation
============

A full installation of the OHBA Software Library includes:

- `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_ (FMRIB Software Library) - only needed if you want to do volumetric source reconstruction.
- `FreeSurfer <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_ (FreeSurfer) - only needed if you want to do surface-based source reconstruction.
- `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_ (or `Anaconda <https://docs.anaconda.com/free/anaconda/install/index.html>`_).
- `osl-ephys <https://github.com/OHBA-analysis/osl-ephys>`_ (OSL Ephys Toolbox).
- `osl-dynamics <https://github.com/OHBA-analysis/osl-dynamics>`_ (OSL Dynamics Toolbox) - only needed if you want to train models for dynamics.

Linux Instructions
------------------

1. Install FSL using the instructions `here <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Linux>`_.

2. Install Freesurfer using the instructions `here <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_.

2. Install `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_ inside the terminal::

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    rm Miniconda3-latest-Linux-x86_64.sh

If you're using a high-performance computing cluster, you may already have :code:`conda` installed as a software module and might be able to load Anaconda with::

    module load Anaconda

and skip step 2.

3. Install osl-ephys and osl-dynamics::

    curl https://raw.githubusercontent.com/OHBA-analysis/osl/main/envs/linux-full.yml > osl.yml
    conda env create -f osl.yml
    rm osl.yml

This will create a conda environment called :code:`osl` which contains both OSL and osl-dynamics.

Mac Instructions
----------------

Instructions:

1. Install FSL using the instructions `here <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/MacOsX>`_.

2. Install `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_ inside the terminal::

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    bash Miniconda3-latest-MacOSX-x86_64.sh
    rm Miniconda3-latest-MacOSX-x86_64.sh

3. Install osl-ephys and osl-dynamics::

    curl https://raw.githubusercontent.com/OHBA-analysis/osl-ephys/main/envs/mac-full.yml > osl.yml
    conda env create -f osl.yml
    rm osl.yml

This will create a conda environment called :code:`osl` which contains both osl-ephys and osl-dynamics.

Windows Instructions
--------------------

If you're using a Windows machine, you will need to install the above in `Ubuntu <https://ubuntu.com/wsl>`_ using a Windows subsystem. 

Instructions:

1. Install FSL using the instructions `here <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation/Windows>`_. Make sure you setup XLaunch for the visualisations.

2. Install `Miniconda <https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html>`_ inside your Ubuntu terminal::

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    rm Miniconda3-latest-Linux-x86_64.sh

3. Install osl-ephys and osl-dynamics::

    curl https://raw.githubusercontent.com/OHBA-analysis/osl-ephys/main/envs/linux-full.yml > osl.yml
    conda env create -f osl.yml
    rm osl.yml

This will create a conda environment called :code:`osl` which contains both osl-ephys and osl-dynamics.

Loading the packages
--------------------

To use osl-ephys/osl-dynamics you need to activate the conda environment::

    conda activate osl

**You need to do every time you open a new terminal.** You know if the :code:`osl` environment is activated if it says :code:`(osl)[...]` at the start of your terminal command line.

Note, if you get a :code:`conda init` error when activating the :code:`osl` environment during a job on an HPC cluster, you can resolve this by replacing::

    conda activate osl

with::

    source activate osl

Integrated Development Environments (IDEs)
------------------------------------------

The OSL installation comes with `Jupyter Notebook <https://jupyter.org/>`_. To open Jupyter Notebook use::

    conda activate osl
    jupyter notebook

There is also an installation with `Sypder <https://www.spyder-ide.org/>`_. To install this on linux use the ``envs/linux-full-with-spyder.yml`` environment. The Mac environments come with Spyder by default. To open Spyder use::

    conda activate osl
    spyder

Test the installation
---------------------

The following should not raise any errors::

    conda activate osl
    python
    >> import osl_ephys
    >> import osl_dynamics

Get the latest source code (optional)
-------------------------------------

If you want the very latest code you can clone the GitHub repo. This is only neccessary if you want recent changes to the package that haven't been released yet.

First install osl-ephys/osl-dynamics using the instructions above. Then clone the repo and install locally from source::

    conda activate osl

    git clone https://github.com/OHBA-analysis/osl-ephys.git
    cd osl-ephys
    pip install -e .
    cd ..

    git clone https://github.com/OHBA-analysis/osl-dynamics.git
    cd osl-dynamics
    pip install -e .

After you install from source, you can run the code with local changes. You can update the source code using::

    git pull

within the :code:`osl-ephys` or :code:`osl-dynamics` directory.

Getting help
------------

If you run into problems while installing OSL, please open an issue on the `GitHub repository <https://github.com/OHBA-analysis/osl-ephys/issues>`_.
