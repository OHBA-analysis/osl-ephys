Installation
============

osl-ephys is on `PyPI <https://pypi.org/project/osl-ephys>`_ and needs Python 3.10 or newer. It runs on Linux and macOS; on Windows, install it inside `Ubuntu on WSL <https://ubuntu.com/wsl>`_ and follow the Linux instructions, as source reconstruction needs FSL, which has no Windows version. We recommend installing osl-ephys into its own conda environment.

Getting conda
-------------

If you do not already have conda, install `Miniforge <https://conda-forge.org/download/>`_, which provides :code:`conda` and :code:`mamba`:

.. code::

    curl -LO "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    rm Miniforge3-$(uname)-$(uname -m).sh

Conda environment (recommended)
-------------------------------

The `envs <https://github.com/OHBA-analysis/osl-ephys/tree/main/envs>`_ directory of the repository has an environment file per platform:

================== ================================= ====================================
Platform           Machine                           Environment file
================== ================================= ====================================
``linux-64``       Linux on x86-64, including        ``linux-64.yml``
                   servers without a display
``linux-aarch64``  Linux on ARM64                    ``linux-aarch64.yml``
``osx-64``         Intel Mac                         ``osx-64.yml``
``osx-arm64``      Apple silicon Mac                 ``osx-arm64.yml``
================== ================================= ====================================

Then download the file and create the environment from it. This fills your platform in for you:

.. code::

    platform=$(conda info | awk '/platform/ {print $3}')
    curl -LO "https://raw.githubusercontent.com/OHBA-analysis/osl-ephys/main/envs/$platform.yml"
    mamba env create -f "$platform.yml"
    conda activate osle
    rm "$platform.yml"

Check the installation with :code:`osl-ephys check`, which imports the package and reports the versions of its main dependencies.

Pip installation
----------------

Alternatively, install everything from PyPI into an empty conda environment:

.. code::

    conda create -n osle python=3.12
    conda activate osle
    pip install osl-ephys

Every dependency ships wheels, so this works on every platform. Prefer the environment files if a compiled dependency gives trouble, e.g. Qt not finding its platform plugin.

FSL and FreeSurfer
------------------

Source reconstruction also needs `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_ for extracting the head surfaces from a structural MRI, and the surface-based route needs `FreeSurfer <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_. Neither is needed for preprocessing.

Headless Linux servers
----------------------

The standard VTK needs a display for the 3D coregistration plots. On Linux x86-64, :code:`linux-64.yml` installs the software-rendering (OSMesa) build of VTK from conda-forge, as the conda package of osl-dynamics does, so the plots render on servers and clusters without a display; set :code:`PYVISTA_OFF_SCREEN=true` in the environment there. This build saves the plots to image and HTML files but cannot open interactive 3D windows, so pass :code:`display=False` (the default) to the plotting functions.

conda-forge has no OSMesa build for ARM Linux, so on an ARM server without a display run osl-ephys under :code:`xvfb-run`, which the Linux distribution's package manager provides.

With a pip installation, swap in the software-rendering wheel instead:

.. code::

    pip uninstall vtk
    pip install --extra-index-url https://wheels.vtk.org vtk-osmesa

Or keep the standard VTK and run osl-ephys under :code:`xvfb-run`.

Install the latest development code (optional)
----------------------------------------------

You should only need to do this if you need a feature or fix that has not been released yet.

In the :code:`osle` environment created above, install the latest development version from the `GitHub repository <https://github.com/OHBA-analysis/osl-ephys>`_ with:

.. code::

    pip install git+https://github.com/OHBA-analysis/osl-ephys.git

Copy the source code (optional)
-------------------------------

To work on osl-ephys itself, install a local copy of the source code (`GitHub repository <https://github.com/OHBA-analysis/osl-ephys>`_) into the :code:`osle` environment:

.. code::

    git clone https://github.com/OHBA-analysis/osl-ephys.git
    cd osl-ephys
    pip install -e .

Now you can run osl-ephys with your own local changes to the code.

Removing osl-ephys
------------------

To remove osl-ephys simply delete the environment:

::

    conda env remove -n osle

The downloaded data files are cached separately, and can be removed with:

::

    rm -rf ~/.cache/osl-files        # linux
    rm -rf ~/Library/Caches/osl-files  # macOS
