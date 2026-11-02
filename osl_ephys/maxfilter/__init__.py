"""MaxFilter: Signal Space Separation of Elekta/MEGIN data with the MaxFilter
command-line tool, which needs a licensed installation.

:func:`run_maxfilter` runs it on a set of files in one of three modes:
``standard`` (one call), ``multistage`` (bad channel detection, then SSS or
tSSS, then an optional head position translation) and ``cbu`` (the MRC
CBU three-stage recipe). Recordings too large for one output file are
processed in chunks and saved as a split chain. In a config it is the
``maxfilter`` section (:mod:`~osl_ephys.maxfilter.steps`), and on the
command line ``osl-ephys maxfilter``.
"""

from osl_ephys.maxfilter.maxfilter import run_maxfilter, output_file
