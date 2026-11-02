"""Data files used by osl-ephys.

This subpackage provides brain atlases, MNI surfaces, masks, scanner layouts
and test recordings needed by the processing pipelines. They are not shipped
with the package: each one is downloaded from
`osl-files <https://github.com/OHBA-analysis/osl-files>`_ on first use and
cached, see :py:mod:`osl_ephys.files._fetch`. The cache is shared with
osl-dynamics, so a file either package has downloaded is used by both. Each
submodule exposes a ``directory`` attribute pointing to its data directory so
files can be resolved by name at runtime using
:py:func:`~osl_ephys.files.functions.check_exists`.

Modules
-------
- :py:mod:`~osl_ephys.files.parcellation` — Volumetric brain parcellations.
- :py:mod:`~osl_ephys.files.mask` — MNI152 brain masks at various resolutions.
- :py:mod:`~osl_ephys.files.surface` — Cortical surface meshes for plotting.
- :py:mod:`~osl_ephys.files.mni152_surfaces` — Pre-extracted MNI152
  skull/scalp surfaces for use with RHINO when no subject MRI is available.
- :py:mod:`~osl_ephys.files.scanner` — MEG scanner layouts and channel name
  files (CTF-275, Neuromag-306).
- :py:mod:`~osl_ephys.files.simulation` — MVAR parameters for simulating
  Neuromag-306 data.
- :py:mod:`~osl_ephys.files.test_data` — Short recordings for the tests.
"""

from osl_ephys.files import (
    mask,
    mni152_surfaces,
    parcellation,
    scanner,
    simulation,
    surface,
    test_data,
)
from osl_ephys.files.functions import *
