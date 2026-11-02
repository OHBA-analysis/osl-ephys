"""Readers and file-finding helpers: SPM M/EEG files, OPM conversion, and
matching session files with wildcards."""

from osl_ephys.io.spm import SPMMEEG
from osl_ephys.io.study import Study
from osl_ephys.io.file_handling import (
    process_file_inputs,
    find_run_id,
    validate_outdir,
    get_rawdir,
    add_subdir,
    sanitise_filepath,
)
