"""Parameters for simulating sensor-space MEG data.

MVAR models fitted to resting-state Neuromag-306 data, reduced with PCA, one
per sensor type (``mag`` and ``grad``): the model parameters, the residual
covariance and the PCA components. Used by
:py:func:`osl_ephys.utils.simulate.simulate_rest_mvar`. These files are
downloaded on first use, see :py:mod:`osl_ephys.files._fetch`.
"""

from osl_ephys.files import _fetch

_SUBDIRECTORY = "simulation"


def file(name: str) -> str:
    """Path to one simulation file, downloading it if needed."""
    return str(_fetch.fetch_file(f"{_SUBDIRECTORY}/{name}"))


def __getattr__(name):
    if name == "path":
        return _fetch.fetch_directory(_SUBDIRECTORY)
    if name == "directory":
        return str(_fetch.fetch_directory(_SUBDIRECTORY))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
