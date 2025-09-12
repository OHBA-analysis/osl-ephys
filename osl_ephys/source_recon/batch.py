"""Batch processing for source reconstruction.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>           

import os
import sys
import traceback
import pprint
import inspect

from copy import deepcopy
from time import localtime, strftime
from functools import partial
from dask.distributed import Variable, Queue

import numpy as np
import yaml
import mne

from . import rhino, wrappers, freesurfer_utils
from ..report import src_report
from ..utils import logger as osl_logger
from ..utils import validate_outdir, find_run_id, parallel
from ..utils.misc import set_random_seed

import logging
logger = logging.getLogger(__name__)


def load_config(config):
    """Load config.

    Parameters
    ----------
    config : str or dict
        Path to yaml file or str to convert to dict or a dict.

    Returns
    -------
    config : dict
        Source reconstruction config.
    """
    if type(config) not in [str, dict]:
        raise ValueError("config must be a str or dict, got {}.".format(type(config)))

    if isinstance(config, str):
        try:
            # See if we have a filepath
            with open(config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            # We have a str
            config = yaml.load(config, Loader=yaml.FullLoader)

    # Validation
    if "source_recon" not in config:
        raise ValueError("source_recon must be included in the config.")

    return config


def find_func(method, extra_funcs):
    """Find a source reconstruction function.

    Parameters
    ----------
    method : str
        Function name.
    extra_funcs : list of functions
        Custom functions.

    Returns
    -------
    func : function
        Function to use.
    """
    func = None

    # Look in custom functions
    if extra_funcs is not None:
        func_ind = [idx if (f.__name__ == method) else -1 for idx, f in enumerate(extra_funcs) ]
        if np.max(func_ind) > -1:
            func = extra_funcs[np.argmax(func_ind)]

    # Look in osl_ephys.source_recon.wrappers
    if func is None and hasattr(wrappers, method):
        func = getattr(wrappers, method)

    return func


def run_src_chain(
    config,
    outdir,
    subject,
    preproc_file=None,
    smri_file=None,
    epoch_file=None,
    surface_extraction_method='fsl',
    logsdir=None,
    reportdir=None,
    gen_report=True,
    verbose="INFO",
    mneverbose="WARNING",
    extra_funcs=None,
    random_seed='auto',
):
    """Source reconstruction.

    Parameters
    ----------
    config : str or dict
        Source reconstruction config.
    outdir : str
        Source reconstruction directory.
    subject : str
        Subject name.
    surface_extraction_method : str
        Can be 'fsl' or 'freesurfer'.
    preproc_file : str
        Preprocessed fif file.
    smri_file : str
        Structural MRI file.
    epoch_file : str
        Epoched fif file.
    logsdir : str
        Directory to save log files to.
    reportdir : str
        Directory to save report files to.
    gen_report : bool
        Should we generate a report?
    verbose : str
        Level of verbose.
    mneverbose : str
        Level of MNE verbose.
    extra_funcs : list of functions
        Custom functions.
    random_seed : 'auto' (default), int or None
        Random seed to set. If 'auto', a random seed will be generated. Random seeds are set for both Python and NumPy.
        If None, no random seed is set.

    Returns
    -------
    flag : bool
        Flag indicating whether source reconstruction was successful.
    """
    if surface_extraction_method == 'fsl':
        rhino.fsl_utils.check_fsl()
    elif surface_extraction_method == 'freesurfer':
        freesurfer_utils.check_freesurfer()

    # Directories
    outdir = validate_outdir(outdir)
    logsdir = validate_outdir(logsdir or outdir / "logs")
    reportdir = validate_outdir(reportdir or outdir / "src_report")
    # Use the subject ID for the run ID
    run_id = subject

    # Generate log filename
    name_base = "{run_id}_{ftype}.{fext}"
    logbase = os.path.join(logsdir, name_base)
    logfile = logbase.format(run_id=run_id, ftype="src", fext="log")
    mne.utils._logging.set_log_file(logfile)

    # Finish setting up loggers
    osl_logger.set_up(prefix=run_id, log_file=logfile, level=verbose, startup=False)
    mne.set_log_level(mneverbose)
    logger = logging.getLogger(__name__)
    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    logger.info("{0} : Starting osl-ephys Processing".format(now))
    logger.info("input : {0}".format(outdir / subject))

    # Set random seed
    if random_seed == 'auto':
        set_random_seed()
    elif random_seed is None:
        pass
    else:
        set_random_seed(random_seed)
    
    # Load config
    if not isinstance(config, dict):
        config = load_config(config)

    # Check what files are in the output directory
    preproc_filename = f"{outdir}/{subject}/{subject}_preproc-raw.fif"
    epoch_filename = f"{outdir}/{subject}/{subject}_epo.fif"
    if os.path.exists(preproc_filename) and os.path.exists(epoch_filename):
        if preproc_file is None and epoch_file is None:
            raise ValueError(
                "Both preproc and epoch fif files found. "
                "Please pass preproc_file=True or epoch_file=True."
            )
    elif os.path.exists(preproc_filename):
        preproc_file = preproc_filename
    elif os.path.exists(epoch_filename):
        epoch_file = epoch_filename

    # Validation
    doing_coreg = (
        any(["compute_surfaces" in method for method in config["source_recon"]]) or
        any(["coregister" in method for method in config["source_recon"]])
    )
    if doing_coreg and smri_file is None:
        raise ValueError("smri_file must be passed if we're doing coregistration.")

    # MAIN BLOCK - Run source reconstruction and catch any exceptions
    try:
        for stage in deepcopy(config["source_recon"]):
            method, userargs = next(iter(stage.items()))
            func = find_func(method, extra_funcs=extra_funcs)
            if func is None:
                avail_funcs = inspect.getmembers(wrappers, inspect.isfunction)
                avail_names = [name for name, _ in avail_funcs]
                if method not in avail_names:
                    raise NotImplementedError(
                        f"{method} not available.\n"
                        + "Please pass via extra_funcs "
                        + f"or use available functions: {avail_names}."
                    )
            def wrapped_func(**kwargs):
                sig = inspect.signature(func)
                args = [param for param in sig.parameters.keys() if param != 'kwargs']
                defaults = [param.default for param in sig.parameters.values() if param.default is not inspect.Parameter.empty]
                args_with_defaults = args[-len(defaults):] if defaults else []
                kwargs_to_pass = {}
                for a in args:
                    if a in kwargs:
                        kwargs_to_pass[a] = kwargs[a]
                    elif a not in args_with_defaults:
                        raise ValueError(f"{a} needs to be passed to {func.__name__}")
                return func(**kwargs_to_pass)
            wrapped_func(
                outdir=outdir,
                subject=subject,
                surface_extraction_method=surface_extraction_method,
                preproc_file=preproc_file,
                smri_file=smri_file,
                epoch_file=epoch_file,
                reportdir=reportdir,
                logsdir=logsdir,
                **userargs,
            )

    except Exception as e:
        logger.critical("*********************************")
        logger.critical("* SOURCE RECONSTRUCTION FAILED! *")
        logger.critical("*********************************")

        ex_type, ex_value, ex_traceback = sys.exc_info()
        logger.error("{0} : {1}".format(method, func))
        logger.error(ex_type)
        logger.error(ex_value)
        logger.error(traceback.print_tb(ex_traceback))

        with open(logfile.replace(".log", ".error.log"), "w") as f:
            f.write("osl-ephys SOURCE RECON failed at: {0}".format(now))
            f.write("\n")
            f.write('Processing failed during stage : "{0}"'.format(method))
            f.write(str(ex_type))
            f.write("\n")
            f.write(str(ex_value))
            f.write("\n")
            traceback.print_tb(ex_traceback, file=f)

        return False

    if gen_report:
        # Generate data and individual HTML data for the report
        src_report.gen_html_data(config, outdir, subject, reportdir, extra_funcs=extra_funcs)

        # Generate individual subject HTML report
        src_report.gen_html_page(reportdir)

    return True


def run_src_batch(
    config,
    outdir,
    subjects,
    preproc_files=None,
    smri_files=None,
    epoch_files=None,
    surface_extraction_method='fsl',
    logsdir=None,
    reportdir=None,
    gen_report=True,
    verbose="INFO",
    mneverbose="WARNING",
    extra_funcs=None,
    dask_client=False,
    random_seed='auto',
):
    """Batch source reconstruction.

    Parameters
    ----------
    config : str or dict
        Source reconstruction config.
    outdir : str
        Source reconstruction directory.
    subjects : list of str
        Subject names.
    surface_extraction_method : str
        Can be 'fsl' or 'freesurfer'.
    preproc_files : list of str
        Preprocessed fif files.
    smri_files : list of str or str
        Structural MRI files. Can be 'standard' to use MNI152_T1_2mm.nii
        for the structural.
    epoch_files : list of str
        Epoched fif file.
    logsdir : str
        Directory to save log files to.
    reportdir : str
        Directory to save report files to.
    gen_report : bool
        Should we generate a report?
    verbose : str
        Level of verbose.
    mneverbose : str
        Level of MNE verbose.
    extra_funcs : list of functions
        Custom functions.
    dask_client : bool
        Are we using a dask client?
    random_seed : 'auto' (default), int or None
        Random seed to set. If 'auto', a random seed will be generated. Random seeds are set for both Python and NumPy.
        If None, no random seed is set.

    Returns
    -------
    flags : list of bool
        Flags indicating whether coregistration was successful.
    """
    if surface_extraction_method == 'fsl':
        rhino.fsl_utils.check_fsl()
    elif surface_extraction_method == 'freesurfer':
        freesurfer_utils.check_freesurfer()

    # Directories
    outdir = validate_outdir(outdir)
    logsdir = validate_outdir(logsdir or outdir / "logs")
    reportdir = validate_outdir(reportdir or outdir / "src_report")

    # Initialise Loggers
    mne.set_log_level(mneverbose)
    logfile = os.path.join(logsdir, 'batch_src.log')
    osl_logger.set_up(log_file=logfile, level=verbose, startup=False)
    logger.info('Starting osl-ephys Batch Source Reconstruction')

    # Set random seed
    if random_seed == 'auto':
        random_seed = set_random_seed()
    elif random_seed is None:
        pass
    else:
        set_random_seed(random_seed)
    
    # Load config
    config = load_config(config)
    config_str = pprint.PrettyPrinter().pformat(config)
    logger.info('Running config\n {0}'.format(config_str))

    # Number of files (subjects) to process
    n_subjects = len(subjects)

    # Validation
    if preproc_files is not None and epoch_files is not None:
        raise ValueError("Please pass either preproc_file or epoch_files, not both.")

    if preproc_files and epoch_files:
        raise ValueError(
            "Cannot pass both preproc_files=True and epoch_files=True. "
            "Please only pass one of these."
        )

    if isinstance(preproc_files, list):
        n_files = len(preproc_files)
        if n_subjects != n_files:
            raise ValueError(f"Got {n_subjects} subjects and {n_files} preproc_files.")

    elif isinstance(epoch_files, list):
        n_files = len(epoch_files)
        if n_subjects != n_files:
            raise ValueError(f"Got {n_subjects} subjects and {n_files} epoch_files.")

    else:
        # Check what files are in the output directory
        preproc_files_list = []
        epoch_files_list = []
        for subject in subjects:
            preproc_file = f"{outdir}/{subject}/{subject}_preproc-raw.fif"
            epoch_file = f"{outdir}/{subject}/{subject}_epo.fif"
            if os.path.exists(preproc_file) and os.path.exists(epoch_file):
                if preproc_files is None and epoch_files is None:
                    raise ValueError(
                        "Both preproc and epoch fif files found. "
                        "Please pass preproc_files=True or epoch_files=True."
                    )
            elif os.path.exists(preproc_file):
                preproc_files_list.append(preproc_file)
            elif os.path.exists(epoch_file):
                epoch_files_list.append(epoch_file)
        if len(preproc_files_list) > 0:
            preproc_files = preproc_files_list
        elif len(epoch_files_list) > 0:
            epoch_files = epoch_files_list

    doing_coreg = (
        any(["compute_surfaces" in method for method in config["source_recon"]]) or
        any(["coregister" in method for method in config["source_recon"]])
    )

    if doing_coreg and smri_files is None:
        raise ValueError("smri_files must be passed if we are coregistering.")
    elif smri_files is None or isinstance(smri_files, str):
        smri_files = [smri_files] * n_subjects

    if preproc_files is None:
        preproc_files = [None] * n_subjects

    if epoch_files is None:
        epoch_files = [None] * n_subjects

    # Create partial function with fixed options
    pool_func = partial(
        run_src_chain,
        surface_extraction_method=surface_extraction_method,
        logsdir=logsdir,
        reportdir=reportdir,
        gen_report=gen_report,
        verbose=verbose,
        mneverbose=mneverbose,
        extra_funcs=extra_funcs,
        random_seed=random_seed,
    )

    # Loop through input files to generate arguments for run_coreg_chain
    args = []
    for subject, preproc_file, smri_file, epoch_file, in zip(subjects, preproc_files, smri_files, epoch_files):
        args.append((config, outdir, subject, preproc_file, smri_file, epoch_file))

    # Actually run the processes
    if dask_client:
        flags = parallel.dask_parallel_bag(pool_func, args)
    else:
        flags = [pool_func(*aa) for aa in args]

    osl_logger.set_up(log_file=logfile, level=verbose, startup=False)
    logger.info("Processed {0}/{1} files successfully".format(int(np.sum(flags)), len(flags)))

    if gen_report and int(np.sum(flags)) > 0:
        # Generate individual subject HTML report
        src_report.gen_html_page(reportdir)

        # Generate a summary report
        if src_report.gen_html_summary(reportdir):
            logger.info("******************************" + "*" * len(str(reportdir)))
            logger.info(f"* REMEMBER TO CHECK REPORT: {reportdir} *")
            logger.info("******************************" + "*" * len(str(reportdir)))

    return flags