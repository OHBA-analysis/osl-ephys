"""File handling utility functions."""

import os
import mne
import csv
import glob
import pathlib
import numpy as np

# Housekeeping for logging
import logging

osl_logger = logging.getLogger(__name__)


def process_file_inputs(inputs):
    """Process inputs for several cases

    The argument, inputs, can be...
    1) string path to unicode file
    2) string path to dir (e.g. if CTF .ds dir)
    3) string path to file or regular-expression matching files
    4) list of string paths to files
    5) list of string paths to dirs (e.g. if CTF .ds dirs)
    6) list of tuples with path to file and output name pairs
    7) list of MNE objects
    """

    infiles = []
    outnames = []
    check_paths = True

    if isinstance(inputs, pathlib.PosixPath):
        inputs = str(inputs)

    process_list = True

    if isinstance(inputs, str):

        # Check if str is a directory path
        if os.path.isdir(inputs):
            # it is a single dir str, put it in a list
            inputs = list([inputs])
        else:
            # assume str is meant to be a file path
            process_list = False
            try:
                # Check if path to unicode file...
                open(inputs, "r")
                infiles, outnames = _load_unicode_inputs(inputs)
            except (UnicodeDecodeError, FileNotFoundError, IndexError):
                # ...else we have a single path or glob expression
                infiles = glob.glob(inputs)
                outnames = [find_run_id(f) for f in infiles]

    if process_list:
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 0:
                raise ValueError("inputs is an empty list!")
            if isinstance(inputs[0], pathlib.PosixPath):
                inputs = [str(i) for i in inputs]
            if isinstance(inputs[0], str):
                # We have a list of paths
                infiles = [sanitise_filepath(f) for f in inputs]
                outnames = [find_run_id(f) for f in infiles]
            elif isinstance(inputs[0], (list, tuple)):
                # We have a list containing files and output names
                for row in inputs:
                    infiles.append(sanitise_filepath(row[0]))
                    outnames.append(row[1])
            elif isinstance(inputs[0], mne.io.Raw):
                # We have a list of MNE objects
                infiles = infiles
                check_paths = False
        else:
            raise ValueError("Input type is invalid")

    # Check that files actually exist if we've been passed filenames rather
    # than objects
    good_files = [1 for ii in range(len(infiles))]
    if check_paths:
        # infiles = [sanitise_filepath(f) for f in infiles]
        for idx, fif in enumerate(infiles):
            if fif.endswith(".ds") or fif.endswith(".mff"):
                good_files[idx] = int(os.path.isdir(fif))
            else:
                good_files[idx] = int(os.path.isfile(fif))
            if good_files[idx] == 0:
                osl_logger.warning("Input file not found: {0}".format(fif))

    if np.all(good_files):
        osl_logger.info("{0} files to be processed.".format(len(infiles)))
    else:
        osl_logger.warning(
            "{0} of {1} input files not found".format(
                np.sum(np.array(good_files) == 0), len(infiles)
            )
        )

    return infiles, outnames, good_files


def sanitise_filepath(fname):
    """Remove leading/trailing whitespace, tab, newline and carriage return
    characters."""
    return fname.strip(" \t\n\r")


def _load_unicode_inputs(fname):
    checked_files = []
    outnames = []
    osl_logger.info("loading inputs from : {0}".format(fname))
    for row in csv.reader(open(fname, "r"), delimiter=","):
        infile = sanitise_filepath(row[0])
        checked_files.append(infile)
        if len(row) > 1:
            outnames.append(row[1])
        else:
            outnames.append(find_run_id(infile))
    return checked_files, outnames


def find_run_id(infile, preload=True):

    # TODO: This is perhaps more complex than it needs to be - could just use
    # the fif option for everything except BTI scans? They're basically the
    # same now.

    if isinstance(infile, mne.io.Raw):
        infile = infile.filenames[0]

    if os.path.split(infile)[1] == "c,rfDC":
        # We have a BTI scan
        runname = os.path.basename(os.path.dirname(infile))
    elif os.path.splitext(infile)[1] == ".fif":
        # We have a FIF file
        # runname = os.path.basename(infile).rstrip('.fif')
        runname = os.path.splitext(os.path.basename(infile))[0]
    elif os.path.splitext(infile)[1] == ".meg4":
        # We have the meg file from a ds directory
        # runname = os.path.basename(infile).rstrip('.ds')
        runname = os.path.splitext(os.path.basename(infile))[0]
    elif os.path.splitext(infile)[1] == ".ds":
        # runname = os.path.basename(infile).rstrip('.ds')
        runname = os.path.splitext(os.path.basename(infile))[0]
    else:
        # Strip to the left of the dot and hope for the best...
        runname = os.path.basename(infile).split(".")[0]
        # raise ValueError('Unable to determine run_id from file {0}'.format(infile))

    return runname


def validate_outdir(outdir):
    """Checks if an output directory exists and if not creates it."""

    outdir = pathlib.Path(outdir)
    if outdir.exists():
        if not os.access(outdir, os.W_OK):
            # Check outdir is a directory
            if not outdir.is_dir():
                raise ValueError("outdir must be the path to a directory.")

            # Check we have write permission
            if not os.access(outdir, os.W_OK):
                raise PermissionError("No write access for {0}".format(outdir))
    else:
        # Output directory doesn't exist
        if outdir.parent.exists():
            # Parent exists, make the output directory
            outdir.mkdir()
        else:
            # Parent doesn't exist
            raise ValueError(
                "Please create the parent directory: {0}".format(outdir.parent)
            )

    return outdir


def get_rawdir(files):
    """Gets the raw data directory from filename(s)."""

    if isinstance(files, list):
        rawfile = pathlib.Path(files[0])
    else:
        rawfile = pathlib.Path(files)

    return rawfile.parent


def add_subdir(file, outdir, run_id=None):
    """Add sub-directory."""

    if not type(outdir) == str:
        outdir = str(outdir)
    if "{" in outdir and "}" in outdir:
        try:
            base = outdir.split("{")[0]
            pat = outdir.split("{")[1].split("}")[0]
            pat0, pat1 = pat.split(":")
            outdir = base + pat0 + file.split(pat0)[1][: int(pat1)]
        except:
            # pattern extraction failed
            raise ValueError(
                "Please make sure the subdirectory structure is present in the input file(s)"
            )
    elif run_id is not None:
        outdir = f"{outdir}/{run_id}"
    return outdir


def osl_print(s, logfile=None):
    print(s)
    if logfile is not None:
        with open(logfile, "a") as f:
            f.write(str(s) + "\n")
