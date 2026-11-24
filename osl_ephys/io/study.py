# Andrew Quinn <a.quinn@bham.ac.uk>
# Mats van Es <mats.vanes@psych.ox.ac.uk>

import re
import glob
import parse
from string import Formatter


class Study:
    """Class for simple file finding and looping.

    Parameters
    ----------
    studydir : str
        The study directory with wildcards.

    Notes
    -----
    This class is a simple wrapper around glob and parse. It works something like this:

    >>> studydir = '/path/to/study/{subject}/{session}/{subject}_{task}.fif'
    >>> study = Study(studydir)

    Get all files in the study directory:

    >>> study.get()

    Get all files for a particular subject:

    >>> study.get(subject='sub-01')

    Get all files for a particular subject and session:

    >>> study.get(subject='sub-01', session='ses-01')

    The fieldnames that are not specified in ``get`` are replaced with wildcards (``*``).
    """

    def __init__(self, studydir):
        """
        Notes
        -----
        This class is a simple wrapper around glob and parse. It works something like this:

        >>> studydir = '/path/to/study/{subject}/{session}/{subject}_{task}.fif'
        >>> study = Study(studydir)

        Get all files in the study directory:

        >>> study.get()

        Get all files for a particular subject:

        >>> study.get(subject='sub-01')

        Get all files for a particular subject and session:

        >>> study.get(subject='sub-01', session='ses-01')

        The fieldnames that are not specified in ``get`` are replaced with wildcards (*).
        """
        self.studydir = studydir

        # Extract field names in between {braces}
        self.fieldnames = [
            fname for _, fname, _, _ in Formatter().parse(self.studydir) if fname
        ]

        # Replace braces with wildcards
        self.globdir = re.sub(r"\{.*?\}", "*", studydir)

        self.match_files = sorted(glob.glob(self.globdir))
        print("found {} files".format(len(self.match_files)))

        self.match_files = [
            ff for ff in self.match_files if parse.parse(self.studydir, ff) is not None
        ]
        print("keeping {} consistent files".format(len(self.match_files)))

        self.match_values = []
        for fname in self.match_files:
            self.match_values.append(parse.parse(self.studydir, fname).named)

        self.fields = {}
        # Use first file as a reference for keywords
        for key, value in self.match_values[0].items():
            self.fields[key] = [value]
            for d in self.match_values[1:]:
                self.fields[key].append(d[key])

    def refresh(self):
        """Refresh the study directory."""
        return self.__init__(self.studydir)

    def get(self, check_exist=True, **kwargs):
        """Get files from the study directory that match the fieldnames.

        Parameters
        ----------
        check_exist : bool
            Whether to check if the files exist.
        **kwargs : dict
            The field names and values to match.

        Returns
        -------
        out : list
            The files that match the field names and values.

        Notes
        -----
        Example using ``Study`` and ``Study.get()``:

        >>> studydir = '/path/to/study/{subject}/{session}/{subject}_{task}.fif'
        >>> study = Study(studydir)

        Get all files in the study directory:

        >>> study.get()

        Get all files for a particular subject:

        >>> study.get(subject='sub-01')

        Get all files for a particular subject and session:

        >>> study.get(subject='sub-01', session='ses-01')

        The fieldnames that are not specified in ``get`` are replaced with wildcards (``*``).
        """
        keywords = {}
        for key in self.fieldnames:
            keywords[key] = kwargs.get(key, "*")

        fname = self.studydir.format(**keywords)

        # we only want the valid files
        if check_exist:
            return [
                ff
                for ff in glob.glob(fname)
                if any(ff in ff_valid for ff_valid in self.match_files)
            ]
        else:
            return glob.glob(fname)
