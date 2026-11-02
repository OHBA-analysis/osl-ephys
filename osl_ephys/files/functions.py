"""Functions used to find the data files osl-ephys downloads from osl-files."""

import os
from os import path


def check_exists(filename: str, directory: str) -> str:
    """Looks for a file in the current working directory and in osl-ephys.

    Parameters
    ----------
    filename : str
        Name of file to look for or a path to a file.
    directory : str
        Path to directory to look in.

    Returns
    -------
    filename : str
        Full path to the file if found.

    Raises
    ------
    FileNotFoundError
        If the file could not be found.
    """
    if not path.exists(filename):

        if path.exists(f"{directory}/{filename}"):
            filename = f"{directory}/{filename}"
        else:
            raise FileNotFoundError(filename)

    return filename


def _find_package_file(filename):
    from osl_ephys.files import mask, parcellation

    """A parcellation or mask from osl-files, by its name or an old osl-ephys name."""
    for directory in (parcellation.directory, mask.directory):
        try:
            return check_exists(filename, directory)
        except FileNotFoundError:
            continue
    return None


def _find_freesurfer_file(filename):
    from mne import label as mne_label

    avail = mne_label._read_annot_cands(
        os.path.join(os.environ["SUBJECTS_DIR"], "fsaverage", "label")
    )
    if filename in avail:
        filename, hemis = mne_label._get_annot_fname(
            None, "fsaverage", "both", filename, os.environ["SUBJECTS_DIR"]
        )
        return filename


def find_file(filename, freesurfer=False):
    """Look for a file: a path, a name in osl-files, or a FreeSurfer annotation.

    Parameters
    ----------
    filename : str
        Path to file to look for.
    freesurfer : bool, optional
        Should we look in the freesurfer directory?

    Returns
    -------
    filename : str
        Path to file found.
    """
    if not os.path.exists(filename):
        if freesurfer:
            filename = _find_freesurfer_file(filename)
        else:
            filename = _find_package_file(filename)

    if filename is None:
        raise FileNotFoundError(filename)

    return filename
