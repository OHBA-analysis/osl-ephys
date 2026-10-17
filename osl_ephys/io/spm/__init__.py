"""Reading SPM M/EEG files into Python.

Loads an SPM ``.mat`` file into an :class:`SPMMEEG` object with the data as
a memory-mapped array and the channel, trial and montage metadata of the
SPM object, for analysing data preprocessed in SPM. It is a reader, not a
converter: the sensor geometry is not guaranteed to match what MNE expects,
so source reconstruction should start from MNE's own formats.

Compiled from scripts by Evan Roberts (https://github.com/evanr70/py_spm)
and Mark Hymers.
"""

from .spmmeeg import SPMMEEG
