"""Short M/EEG recordings for testing.

One per scanner type, with the geometry of a real recording and a synthetic
signal, see the README in the ``test_data`` directory of osl-files. They are
downloaded on first use, see :py:mod:`osl_ephys.files._fetch`.
"""

from osl_ephys.files import _fetch

_SUBDIRECTORY = "test_data"

_FILES = {
    "elekta": "elekta/rest-raw.fif",
    "ctf": "ctf/rest-raw.fif",
    "ctf_headshape": "ctf/headshape.pos",
    "opm": "opm/task-raw.fif",
    "eeg": "eeg/rest.vhdr",
    "eeg_montage": "eeg/rest-dig.fif",
}


def file(name: str) -> str:
    """Path to one test file, downloading it if needed."""
    return str(_fetch.fetch_file(f"{_SUBDIRECTORY}/{name}"))


def __getattr__(name):
    if name == "eeg":
        # BrainVision is three files; fetch the data and markers with the header
        for ext in (".vmrk", ".eeg"):
            _fetch.fetch_file(f"{_SUBDIRECTORY}/eeg/rest{ext}")
    if name in _FILES:
        return file(_FILES[name])
    if name == "path":
        return _fetch.fetch_directory(_SUBDIRECTORY)
    if name == "directory":
        return str(_fetch.fetch_directory(_SUBDIRECTORY))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
