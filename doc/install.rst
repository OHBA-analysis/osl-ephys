Installation
============

osl-ephys is on `PyPI <https://pypi.org/project/osl-ephys>`_ and needs Python 3.10 or newer. We recommend installing it into its own conda environment, with `Miniforge <https://conda-forge.org/download/>`_ if you do not already have conda:

.. code::

    conda create -n osle python=3.12
    conda activate osle
    pip install osl-ephys

Check the installation with :code:`osl-ephys check`, which imports the package and reports the versions of its main dependencies.

Source reconstruction also needs `FSL <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FslInstallation>`_ for extracting the head surfaces from a structural MRI, and the surface-based route needs `FreeSurfer <https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall>`_. Neither is needed for preprocessing. On Windows, install everything inside `Ubuntu on WSL <https://ubuntu.com/wsl>`_.

Headless Linux servers
----------------------

The VTK wheel on PyPI needs a display for the 3D coregistration plots. On a server without one, either run osl-ephys under :code:`xvfb-run`, or swap in the software-rendering build of VTK:

.. code::

    pip uninstall vtk
    pip install --extra-index-url https://wheels.vtk.org vtk-osmesa

and set :code:`PYVISTA_OFF_SCREEN=true` in the environment.

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
