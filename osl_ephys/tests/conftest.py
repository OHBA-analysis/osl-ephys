"""Test configuration.

No plot windows, in the tests or in the workers they spawn; the test
recordings fetched once per session; the markers of the tests that need
FSL or FreeSurfer, skipped unless the installation is found.
"""

import os

import matplotlib
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")
matplotlib.use("Agg")


@pytest.fixture(scope="session")
def recordings():
    """The four test recordings (fetched from osl-files on first use), by name."""
    from osl_ephys import files

    return {
        name: str(getattr(files.test_data, name))
        for name in ("elekta", "ctf", "opm", "eeg")
    }


@pytest.fixture(scope="session")
def two_sessions(recordings):
    """A manifest of two sessions of the same subject."""
    return [
        {"id": "elekta", "subject": "sub-01", "raw_file": recordings["elekta"]},
        {"id": "elekta2", "subject": "sub-01", "raw_file": recordings["elekta"]},
    ]


def pytest_runtest_setup(item):
    if "fsl" in item.keywords and not os.environ.get("FSLDIR"):
        pytest.skip("FSL is not installed (FSLDIR is not set)")
    if "freesurfer" in item.keywords and not os.environ.get("FREESURFER_HOME"):
        pytest.skip("FreeSurfer is not installed (FREESURFER_HOME is not set)")
