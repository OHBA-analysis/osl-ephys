#!/usr/bin/env python

"""Tools for batch preprocessing.

"""

# Authors: Andrew Quinn <a.quinn@bham.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es  <mats.vanes@psych.ox.ac.uk>

import argparse
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import pprint
import traceback
import re
import logging
import pickle
from pathlib import Path
from copy import deepcopy
from functools import partial, wraps
from time import localtime, strftime
from datetime import datetime
import inspect

import mne
import numpy as np
import yaml

from . import mne_wrappers, osl_wrappers
from ..utils import find_run_id, validate_outdir, process_file_inputs
from ..utils import logger as osl_logger
from ..utils.parallel import dask_parallel_bag
from ..utils.version_utils import check_version
from ..utils.misc import set_random_seed

logger = logging.getLogger(__name__)


# --------------------------------------------------------------
# Decorators


def print_custom_func_info(func):
    """Prints info for user-specified functions.

    Parameters
    ----------
    func : function
        Function to wrap.

    Returns
    -------
    function
        Wrapped function.
    """

    @wraps(func)
    def wrapper(dataset, userargs):
        logger.info("CUSTOM Stage - {}".format(func.__name__))
        logger.info("userargs: {}".format(str(userargs)))
        return func(dataset, userargs)

    return wrapper


# --------------------------------------------------------------
# Data importers


def import_data(infile, preload=True):
    """Imports data from a file.

    Parameters
    ----------
    infile : str
        Path to file to read. File can be bti, fif, ds, meg4 or vhdr.
    preload : bool
        Should we load the data in the file?

    Returns
    -------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        Data as an MNE Raw object.
    """
    if not isinstance(infile, str):
        raise ValueError(
            "infile must be a str. Got type(infile)={0}.".format(type(infile))
        )
    if " " in infile:
        raise ValueError("filename cannot contain spaces.")

    logger.info("IMPORTING: {0}".format(infile))

    # BTI scan
    if os.path.split(infile)[1] == "c,rfDC":
        logger.info("Detected BTI file format, using: mne.io.read_raw_bti")
        if os.path.isfile(os.path.join(os.path.split(infile)[0], "hs_file")):
            head_shape_fname = "hs_file"
        else:
            head_shape_fname = None
        raw = mne.io.read_raw_bti(infile, head_shape_fname=head_shape_fname, preload=preload)

    # FIF file
    elif os.path.splitext(infile)[1] == ".fif":
        logger.info("Detected fif file format, using: mne.io.read_raw_fif")
        raw = mne.io.read_raw_fif(infile, preload=preload)

    # EDF file
    elif os.path.splitext(infile)[1].lower() == ".edf":
        logger.info("Detected edf file format, using: mne.io.read_raw_edf")
        raw = mne.io.read_raw_edf(infile, preload=preload)

    # CTF data in ds directory
    elif os.path.splitext(infile)[1] == ".ds":
        logger.info("Detected CTF file format, using: mne.io.read_raw_ctf")
        raw = mne.io.read_raw_ctf(infile, preload=preload)

    elif os.path.splitext(infile)[1] == ".meg4":
        logger.info("Detected CTF file format, using: mne.io.read_raw_ctf")
        raw = mne.io.read_raw_ctf(os.path.dirname(infile), preload=preload)

    # Brainvision
    elif os.path.splitext(infile)[1] == ".vhdr":
        logger.info("Detected brainvision file format, using: mne.io.read_raw_brainvision")
        raw = mne.io.read_raw_brainvision(infile, preload=preload)

    # EEGLAB .set
    elif os.path.splitext(infile)[1] == ".set":
        logger.info("Detected EEGLAB file format, using: mne.io.read_raw_eeglab")
        raw = mne.io.read_raw_eeglab(infile, preload=preload)

    elif os.path.splitext(infile)[1] == ".con" or os.path.splitext(infile)[1] == ".sqd":
        logger.info("Detected Ricoh/KIT file format, using: mne.io.read_raw_kit")
        raw = mne.io.read_raw_kit(infile, preload=preload)

    elif os.path.splitext(infile)[1] == ".bdf":
        logger.info("Detected BDF file format, using: mne.io.read_raw_bdf")
        raw = mne.io.read_raw_bdf(infile, preload=preload)

    elif os.path.splitext(infile)[1] == ".mff":
        logger.info("Detected EGI file format, using mne.io.read_raw_egi")
        raw = mne.io.read_raw_egi(infile, preload=preload)
        
    # Other formats not accepted
    else:
        msg = "Unable to determine file type of input {0}".format(infile)
        logger.error(msg)
        raise ValueError(msg)

    return raw


# --------------------------------------------------------------
# Batch processing utilities


def find_func(method, target="raw", extra_funcs=None):
    """Find a preprocessing function.

    Function priority:

    1. User custom function
    
    2. MNE/osl-ephys wrapper
    
    3. MNE method on Raw or Epochs (specified by target)

    Parameters
    ----------
    method : str
        Function name.
    target : str
        Type of MNE object to preprocess. Can be ``'raw'``, ``'epochs'``, ``'evoked'``, ``'power'`` or ``'itc'``.
    extra_funcs : list
        List of user-defined functions.

    Returns
    -------
    function
        Function to preprocess an MNE object.
    """
    func = None

    # 1) user custom function
    if extra_funcs is not None:
        func_ind = [
            idx if (f.__name__ == method) else -1 for idx, f in enumerate(extra_funcs)
        ]
        if np.max(func_ind) > -1:
            func = extra_funcs[np.argmax(func_ind)]
            func = print_custom_func_info(func)

    # 2) MNE/osl-ephys Wrapper
    # Find osl-ephys function in local module
    if func is None and hasattr(osl_wrappers, "run_osl_{}".format(method)):
        func = getattr(osl_wrappers, "run_osl_{}".format(method))

    # Find MNE function in local module
    if func is None and hasattr(mne_wrappers, "run_mne_{}".format(method)):
        func = getattr(mne_wrappers, "run_mne_{}".format(method))

    # 3) MNE direct method
    if func is None:
        if target == "raw":
            if hasattr(mne.io.Raw, method) and callable(getattr(mne.io.Raw, method)):
                func = partial(mne_wrappers.run_mne_anonymous, method=method)
        elif target == "epochs":
            if hasattr(mne.Epochs, method) and callable(getattr(mne.Epochs, method)):
                func = partial(mne_wrappers.run_mne_anonymous, method=method)
        elif target in ("power", "itc"):
            if hasattr(mne.time_frequency.EpochsTFR, method) and callable(
                getattr(mne.time_frequency.EpochsTFR, method)
            ):
                func = partial(mne_wrappers.run_mne_anonymous, method=method)

    if func is None:
        logger.critical("Function not found! {}".format(method))

    return func 


def load_config(config):
    """Load config.

    Parameters
    ----------
    config : str or dict
        Path to yaml file or string to convert to dict or a dict.

    Returns
    -------
    dict
        Preprocessing config.
    """
    if type(config) not in [str, dict]:
        raise ValueError("config must be a str or dict, got {}.".format(type(config)))

    if isinstance(config, str):
        try:
            # See if we have a filepath
            with open(config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            # We have a string
            config = yaml.load(config, Loader=yaml.FullLoader)

    # do some checks on the config
    for key in config:
        if config[key] == 'None':
            config[key] = None
    
    # Initialise missing values in config
    if "meta" not in config:
        config["meta"] = {"event_codes": None}
    elif "event_codes" not in config["meta"]:
        config["meta"]["event_codes"] = None
    elif "versions" not in config['meta']:
        config["meta"]["versions"] = None

    if "preproc" not in config and "group" not in config:
        raise KeyError("Please specify preprocessing and/or group processing steps in config.")

    if "preproc" in config and config["preproc"] is not None:
        for stage in config["preproc"]:
            # Check each stage is a dictionary with a single key
            if not isinstance(stage, dict):
                raise ValueError(
                    "Preprocessing stage '{0}' is a {1} not a dict".format(
                        stage, type(stage)
                    )
                )

            if len(stage) != 1:
                raise ValueError(
                    "Preprocessing stage '{0}' should only have a single key".format(stage)
                )

            for key, val in stage.items():
                # internally we want options to be an empty dict (for now at least)
                if val in ["null", "None", None]:
                    stage[key] = {}

        for step in config["preproc"]:
            if config["meta"]["event_codes"] is None and "find_events" in step.values():
                raise KeyError(
                    "event_codes must be passed in config if we are finding events."
                )
    else:
        config['preproc'] = None
                
    if "group" in config and config["group"] is not None:
        for stage in config["group"]:
            # Check each stage is a dictionary with a single key
            if not isinstance(stage, dict):
                raise ValueError(
                    "Group processing stage '{0}' is a {1} not a dict".format(
                        stage, type(stage)
                    )
                )

            if len(stage) != 1:
                raise ValueError(
                    "Group processing stage '{0}' should only have a single key".format(stage)
                )

            for key, val in stage.items():
                # internally we want options to be an empty dict (for now at least)
                if val in ["null", "None", None]:
                    stage[key] = {}
    else:
        config['group'] = None

    return config


def check_config_versions(config):
    """Get config from a preprocessed fif file.

    Parameters
    ----------
    config : dictionary or yaml string
        Preprocessing configuration to check.

    Raises
    ------
    AssertionError
        Raised if package version mismatch found in 'version_assert'
    Warning
        Raised if package version mismatch found in 'version_warn'
    """
    config = load_config(config)

    # Check for version and raise an error if mismatch found
    if 'version_assert' in config['meta']:
        for vers in config['meta']['version_assert']:
            check_version(vers, mode='assert')

    # Check for version and raise a warning if mismatch found
    if 'version_warn' in config['meta']:
        for vers in config['meta']['version_warn']:
            check_version(vers, mode='warn')


def get_config_from_fif(inst):
    """Get config from a preprocessed fif file.

    Reads the ``inst.info['description']`` field of a fif file to get the preprocessing config.
    
    Parameters
    ----------
    inst : :py:class:`mne.io.Raw <mne.io.Raw>`, :py:class:`mne.Epochs <mne.Epochs>`, :py:class:`mne.Evoked <mne.Evoked>`
        Preprocessed MNE object.

    Returns
    -------
    dict
        Preprocessing config.
    """
    config_list = re.findall(
        "%% config start %%(.*?)%% config end %%",
        inst.info["description"],
        flags=re.DOTALL,
    )
    config = []
    for config_text in config_list:
        config.append(load_config(config_text))

    return config


def append_preproc_info(dataset, config, extra_funcs=None):
    """Add to the config of already preprocessed data to ``inst.info['description']``.

    Parameters
    ----------
    dataset : dict
        Preprocessed dataset.
    config : dict
        Preprocessing config.

    Returns
    -------
    dict
        Dataset dict containing the preprocessed data edited in place.
    """
    from .. import __version__  # here to avoid circular import

    if dataset["raw"].info["description"] == None:
        dataset["raw"].info["description"] = ""

    preproc_info = (
        "\n\nOSL-EPHYS BATCH PROCESSING APPLIED ON "
        + f"{datetime.today().strftime('%d/%m/%Y %H:%M:%S')} \n"
        + f"VERSION: {__version__}\n"
        + f"%% config start %% \n{config} \n%% config end %%"
    )
    
    if extra_funcs is not None:
        preproc_info += "\n\nCUSTOM FUNCTIONS USED:\n"
        for func in extra_funcs:
            preproc_info += f"%% extra_funcs start %% \n{inspect.getsource(func)}\n%% extra_funcs end %%"
    
    dataset["raw"].info["description"] = (
        dataset["raw"].info["description"] + preproc_info
    )

    if dataset["epochs"] is not None:
        if dataset["epochs"].info["description"] == None:
            dataset["epochs"].info["description"] = ""
        dataset["epochs"].info["description"] = (
            dataset["epochs"].info["description"] + preproc_info
        )

    return dataset


def write_dataset(dataset, outbase, run_id, ftype='preproc-raw', overwrite=False, skip=None):
    """Write preprocessed data to a file.

    Will write all keys in the dataset dict to disk with corresponding extensions.

    Parameters
    ----------
    dataset : dict
        Preprocessed dataset.
    outbase : str
        Path to directory to write to.
    run_id : str
        ID for the output file.
    ftype: str
        Extension for the fif file (default ``preproc-raw``)
    overwrite : bool
        Should we overwrite if the file already exists?
    skip : list or None
        List of keys to skip writing to disk. If None, we don't skip any keys.

    Output
    ------
    fif_outname : str
        The saved fif file name
    """
    
    if skip is None:
        skip = []
    else:
        [logger.info("Skip saving of dataset['{}']".format(key)) for key in skip]

    # Strip "_preproc-raw" or "_raw" from the run id
    for string in ["_preproc-raw", "_raw"]:
        if string in run_id:
            run_id = run_id.replace(string, "")
    
    if "raw" in skip:
        outnames = {"raw": None}
    else:
        outnames = {"raw": outbase.format(run_id=run_id, ftype=ftype, fext="fif")}
        if Path(outnames["raw"]).exists() and not overwrite:
            raise ValueError(
                "{} already exists. Please delete or do use overwrite=True.".format(outnames['raw'])
            )
        logger.info(f"Saving dataset['raw'] as {outnames['raw']}")
        dataset["raw"].save(outnames['raw'], overwrite=overwrite)   

    if "events" in dataset and "events" not in skip and dataset['events'] is not None:
        outnames['events'] = outbase.format(run_id=run_id, ftype="events", fext="npy")
        logger.info(f"Saving dataset['events'] as {outnames['events']}")
        np.save(outnames['events'], dataset["events"])

    if "event_id" in dataset and "event_id" not in skip and dataset['event_id'] is not None:
        outnames['event_id'] = outbase.format(run_id=run_id, ftype="event-id", fext="yml")
        logger.info(f"Saving dataset['event_id'] as {outnames['event_id']}")
        yaml.dump(dataset["event_id"], open(outnames['event_id'], "w"))

    if "epochs" in dataset and "epochs" not in skip and dataset['epochs'] is not None:
        outnames['epochs'] = outbase.format(run_id=run_id, ftype="epo", fext="fif")
        logger.info(f"Saving dataset['epochs'] as {outnames['epochs']}")
        dataset["epochs"].save(outnames['epochs'], overwrite=overwrite)

    if "ica" in dataset and "ica" not in skip and dataset['ica'] is not None:
        outnames['ica'] = outbase.format(run_id=run_id, ftype="ica", fext="fif")
        logger.info(f"Saving dataset['ica'] as {outnames['ica']}")
        dataset["ica"].save(outnames['ica'], overwrite=overwrite)

    if "tfr" in dataset and "tfr" not in skip and dataset['tfr'] is not None:
        outnames['tfr'] = outbase.format(run_id=run_id, ftype="tfr", fext="fif")
        logger.info(f"Saving dataset['tfr'] as {outnames['tfr']}")
        dataset["tfr"].save(outnames['tfr'], overwrite=overwrite)

    if "glm" in dataset and "glm" not in skip and dataset['glm'] is not None:
        outnames['glm'] = outbase.format(run_id=run_id, ftype="glm", fext="pkl")
        logger.info(f"Saving dataset['glm'] as {outnames['glm']}")
        dataset["glm"].save_pkl(outnames['glm'], overwrite=overwrite)
    
    if "fig" in dataset and "fig" not in skip and dataset['fig'] is not None:
        keys = dataset["fig"].keys()
        outnames['fig'] = {}
        for key in keys:
            outnames['fig'][key] = outbase.format(run_id=run_id, ftype=key, fext="png")
            logger.info(f"Saving dataset['fig'][{key}] as {outnames['fig'][key]}")
            dataset["fig"][key].savefig(outnames['fig'][key])
            
    # save remaining keys as pickle files
    for key in dataset:
        if key not in outnames and key not in skip:
            outnames[key] = outbase.format(run_id=run_id, ftype=key, fext="pkl")
            logger.info(f"Saving dataset['{key}'] as {outnames[key]}")
            if (not os.path.exists(outnames[key]) or overwrite) and key not in skip and dataset[key] is not None:
                with open(outnames[key], "wb") as f:
                    pickle.dump(dataset[key], f)
    return outnames


def read_dataset(fif, preload=False, ftype=None):
    """Reads ``fif``/``npy``/``yml`` files associated with a dataset.

    Parameters
    ----------
    fif : str
        Path to raw fif file (can be preprocessed).
    preload : bool
        Should we load the raw fif data?
    ftype : str
        Extension for the fif file (will be replaced for e.g. ``'_events.npy'`` or 
        ``'_ica.fif'``). If ``None``, we assume the fif file is preprocessed with 
        ``osl-ephys`` and has the extension ``'_preproc-raw'``. If this fails, we guess 
        the extension as whatever comes after the last ``'_'``.

    Returns
    -------
    dataset : dict
        Contains keys: ``'raw'``, ``'events'``, ``'event_id'``, ``'epochs'``, ``'ica'``.
    """
    print("Loading dataset:")

    print("Reading", fif)
    raw = mne.io.read_raw_fif(fif, preload=preload)

    # Guess extension
    if ftype is None:
        logger.info("Guessing the preproc extension")
        if "preproc-raw" in fif:
            logger.info('Assuming fif file type is "preproc-raw"')
            ftype = "preproc-raw"
        else:
            if len(fif.split("_"))<2:
                logger.error("Unable to guess the fif file extension")
            else:
                logger.info('Assuming fif file type is the last "_" separated string')
                ftype = fif.split("_")[-1].split('.')[-2]
    
    # add extension to fif file name
    ftype = ftype + ".fif"
    
    events = Path(fif.replace(ftype, "events.npy"))
    if events.exists():
        print("Reading", events)
        events = np.load(events)
    else:
        events = None

    event_id = Path(fif.replace(ftype, "event-id.yml"))
    if event_id.exists():
        print("Reading", event_id)
        with open(event_id, "r") as file:
            event_id = yaml.load(file, Loader=yaml.Loader)
    else:
        event_id = None

    epochs = Path(fif.replace(ftype, "epo.fif"))
    if epochs.exists():
        print("Reading", epochs)
        epochs = mne.read_epochs(epochs)
    else:
        epochs = None

    ica = Path(fif.replace(ftype, "ica.fif"))
    if ica.exists():
        print("Reading", ica)
        ica = mne.preprocessing.read_ica(ica)
    else:
        ica = None

    dataset = {
        "raw": raw,
        "events": events,
        "event_id": event_id,
        "epochs": epochs,
        "ica": ica,
    }

    return dataset


def plot_preproc_flowchart(
    config,
    outname=None,
    show=True,
    stagecol="wheat",
    startcol="red",
    fig=None,
    ax=None,
    title=None,
):
    """Make a summary flowchart of a preprocessing chain.

    Parameters
    ----------
    config : dict
        Preprocessing config to plot.
    outname : str
        Output filename.
    show : bool
        Should we show the plot?
    stagecol : str
        Stage colour.
    startcol : str
        Start colour.
    fig : matplotlib.figure
        Matplotlib figure to plot on.
    ax : :py:class:`matplotlib.axes <matplotlib.axes>`
        Matplotlib axes to plot on.
    title : str
        Title for the plot.

    Returns
    -------
    fig : :py:class:`matplotlib.figure <matplotlib.figure>`
    ax : :py:class:`matplotlib.axes <matplotlib.axes>`
    """
    config = load_config(config)

    if np.logical_or(ax == None, fig == None):
        fig = plt.figure(figsize=(8, 12))
        plt.subplots_adjust(top=0.95, bottom=0.05)
        ax = plt.subplot(111, frame_on=False)
    ax.set_xticks([])
    ax.set_yticks([])
    if title == None:
        ax.set_title("osl-ephys Processing Recipe", fontsize=24)
    else:
        ax.set_title(title, fontsize=24)
    
    tmp_h = 1
    if config["preproc"] is not None:
        tmp_h += 1 + len(config["preproc"])
    if config["group"] is not None:
        tmp_h += 1 + len(config["group"])
    stage_height = 1 / tmp_h

    box = dict(boxstyle="round", facecolor=stagecol, alpha=1, pad=0.3)
    startbox = dict(boxstyle="round", facecolor=startcol, alpha=1)
    font = {
        "family": "serif",
        "color": "k",
        "weight": "normal",
        "size": 16,
    }
    stages = [{"input": ""}]
    if config['preproc'] is not None:
        stages += [{"preproc": ""}, *config["preproc"]]
    if config['group'] is not None:
        stages += [{"group": ""}, *config["group"]]
    stages.append({"output": ""})
    stage_str = "$\\bf{{{0}}}$ {1}"

    ax.arrow(
        0.5, 1, 0.0, -1+0.02, fc="k", ec="k", head_width=0.045,
        head_length=0.035, length_includes_head=True,
    )

    for idx, stage in enumerate(stages):
        method, userargs = next(iter(stage.items()))

        method = method.replace("_", "\_")
        if method in ["input", "preproc", "group", "output"]:
            b = startbox
        else:
            b = box
            method = method + ":"

        ax.text(
            0.5,
            1 - stage_height * idx,
            stage_str.format(method, str(userargs)[1:-1]),
            ha="center",
            va="center",
            bbox=b,
            fontdict=font,
            wrap=True,
        )

    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.25, 0.75)

    if outname is not None:
        fig.savefig(outname, dpi=300, transparent=True)

    if show is True:
        fig.show()

    return fig, ax


# --------------------------------------------------------------
# Batch processing


def run_proc_chain(
    config,
    infile,
    subject=None,
    ftype='preproc-raw',
    outdir=None,
    logsdir=None,
    reportdir=None,
    ret_dataset=True,
    gen_report=None,
    overwrite=False,
    skip_save=None,
    extra_funcs=None,
    random_seed='auto',
    verbose="INFO",
    mneverbose="WARNING",
):
    """Run preprocessing for a single file.

    Parameters
    ----------
    config : str or dict
        Preprocessing config.
    infile : str
        Path to input file.
    subject : str
        Subject ID. This will be the sub-directory in outdir.
    ftype: str
        Extension for the fif file (default ``preproc-raw``)
    outdir : str
        Output directory.
    logsdir : str
        Directory to save log files to.
    reportdir : str
        Directory to save report files to.
    ret_dataset : bool
        Should we return a dataset dict?
    gen_report : bool
        Should we generate a report?
    overwrite : bool
        Should we overwrite the output file if it already exists?
    skip_save: list or None (default)
        List of keys to skip writing to disk. If None, we don't skip any keys.
    extra_funcs : list
        User-defined functions.
    random_seed : 'auto' (default), int or None
        Random seed to set. If 'auto', a random seed will be generated. Random seeds are set for both Python and NumPy.
        If None, no random seed is set.
    verbose : str
        Level of info to print.
        Can be: ``'CRITICAL'``, ``'ERROR'``, ``'WARNING'``, ``'INFO'``, ``'DEBUG'`` or ``'NOTSET'``.
    mneverbose : str
        Level of info from MNE to print.
        Can be: ``'CRITICAL'``, ``'ERROR'``, ``'WARNING'``, ``'INFO'``, ``'DEBUG'`` or ``'NOTSET'``.

    Returns
    -------
    dict or bool
        If ``ret_dataset=True``, a dict containing the preprocessed dataset with the following keys: ``raw``, ``ica``, ``epochs``, ``events``, ``event_id``.
        An empty dict is returned if preprocessing fails. If ``ret_dataset=False``, we return a flag indicating whether preprocessing was successful.
    """

    # Get run (subject) ID
    run_id = subject or find_run_id(infile)
    name_base = "{run_id}_{ftype}.{fext}"

    if not ret_dataset:
        # Let's make sure we have an output directory
        outdir = outdir or os.getcwd()

    if outdir is not None:
        # We're saving the output to disk

        # Generate a report by default, this is overriden if the user passes
        # gen_report=False
        gen_report = True if gen_report is None else gen_report
        
        # Create output directories if they don't exist
        outdir = validate_outdir(outdir)
        logsdir = validate_outdir(logsdir or outdir / "logs")
        reportdir = validate_outdir(reportdir or outdir / "preproc_report")
        outdir = validate_outdir(outdir / run_id)

    else:
        # We're not saving the output to disk

        # Don't generate a report by default, this is overriden if the user passes
        # something for reportdir or gen_report=True
        gen_report = gen_report or reportdir is not None or False
        if gen_report:
            # Make sure we have a directory to write the report to
            reportdir = validate_outdir(reportdir or os.getcwd() + "/preproc_report")

        # Allow the user to create a log if they pass logsdir
        if logsdir is not None:
            logsdir = validate_outdir(logsdir)

    # Create output filename
    if outdir is not None:
        outbase = os.path.join(outdir, name_base)

    # Generate log filename
    if logsdir is not None:
        logbase = os.path.join(logsdir, name_base)
        logfile = logbase.format(run_id=run_id, ftype=ftype.replace("-raw", ""), fext="log")
        mne.utils._logging.set_log_file(logfile, overwrite=overwrite)
    else:
        logfile = None

    # Finish setting up loggers
    osl_logger.set_up(prefix=run_id, log_file=logfile, level=verbose, startup=False)
    mne.set_log_level(mneverbose)
    logger = logging.getLogger(__name__)
    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    logger.info("{0} : Starting osl-ephys Processing".format(now))
    logger.info("input : {0}".format(infile))

    # Set random seed
    if random_seed == 'auto':
        set_random_seed()
    elif random_seed is None:
        pass
    else:
        set_random_seed(random_seed)
    
    # Write preprocessed data to output directory
    if outdir is not None:
        # Check for existing outputs - should be a .fif at least
        fifout = outbase.format(run_id=run_id, ftype=ftype, fext='fif')
        if os.path.exists(fifout) and (overwrite is False):
            logger.critical('Skipping preprocessing - existing output detected')
            return False

    # Load config
    if not isinstance(config, dict):
        config = load_config(config)

    # MAIN BLOCK - Run the preproc chain and catch any exceptions
    try:
        if isinstance(infile, str):
            raw = import_data(infile)
        elif (isinstance(infile, mne.io.fiff.raw.Raw)
              or isinstance(infile, mne.io.curry.curry.RawCurry)):
            raw = infile
            infile = raw.filenames[0]  # assuming only one file here

        # Create a dataset dict to hold the preprocessed dataset
        dataset = {
            "raw": raw,
            "events": None,
            "epochs": None,
            "event_id": config["meta"]["event_codes"],
            "ica": None,
            "fig": {},
        }

        # Do the preprocessing
        for stage in deepcopy(config["preproc"]):
            method, userargs = next(iter(stage.items()))
            target = userargs.get("target", "raw")  # Raw is default
            func = find_func(method, target=target, extra_funcs=extra_funcs)
            
            # Actual function call
            dataset = func(dataset, userargs)

        # Add preprocessing info to dataset dict
        dataset = append_preproc_info(dataset, config, extra_funcs)

        outnames = {"raw": None}
        if outdir is not None:
            outnames = write_dataset(dataset, outbase, run_id, overwrite=overwrite, skip=skip_save)

        # Generate report data
        if gen_report:
            # Switch to non-GUI plotting backend
            mpl_backend = matplotlib.pyplot.get_backend()
            matplotlib.use('Agg')

            from ..report import gen_html_data, gen_html_page  # avoids circular import
            logger.info("{0} : Generating Report".format(now))
            report_data_dir = validate_outdir(reportdir / Path(outnames["raw"]).stem)
            gen_html_data(
                dataset["raw"],
                report_data_dir,
                ica=dataset["ica"],
                preproc_fif_filename=outnames["raw"],
                logsdir=logsdir,
                run_id=run_id,
            )
            gen_html_page(reportdir)

            # Restore plotting context
            matplotlib.use(mpl_backend)

    except Exception as e:
        # Preprocessing failed

        if 'method' not in locals():
            method = 'import_data'
            func = import_data

        logger.critical("**********************")
        logger.critical("* PROCESSING FAILED! *")
        logger.critical("**********************")

        ex_type, ex_value, ex_traceback = sys.exc_info()
        logger.error("{0} : {1}".format(method, func))
        logger.error(ex_type)
        logger.error(ex_value)
        logger.error(traceback.print_tb(ex_traceback))

        with open(logfile.replace(".log", ".error.log"), "w") as f:
            f.write("OSL-EPHYS PREPROCESSING CHAIN FAILED AT: {0}".format(now))
            f.write("\n")
            f.write('Processing failed during stage : "{0}"'.format(method))
            f.write(str(ex_type))
            f.write("\n")
            f.write(str(ex_value))
            f.write("\n")
            traceback.print_tb(ex_traceback, file=f)

        if ret_dataset:
            # We return an empty dict to indicate preproc failed
            # This ensures the function consistently returns one
            # variable type
            return {}
        else:
            if 'group' in config:
                return False, None
            return False

    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    logger.info("{0} : Processing Complete".format(now))

    if outnames["raw"] is not None:
        logger.info("Output file is {}".format(outnames["raw"]))

    if ret_dataset:
        return dataset
    else:
        if 'group' in config:
            return True, outnames
        return True


def run_proc_batch(
    config,
    files,
    subjects=None,
    ftype='preproc-raw',
    outdir=None,
    logsdir=None,
    reportdir=None,
    gen_report=True,
    overwrite=False,
    skip_save=None,
    extra_funcs=None,
    covs=None,
    random_seed='auto',
    verbose="INFO",
    mneverbose="WARNING",
    strictrun=False,
    dask_client=False,
):
    """Run batched preprocessing.

    This function will write output to disk (i.e. will not return the preprocessed
    data).

    Parameters
    ----------
    config : str or dict
        Preprocessing config.
    files : str or list or mne.Raw
        Can be a list of Raw objects or a list of filenames (or ``.ds`` dir names if CTF data)
        or a path to a textfile list of filenames (or ``.ds`` dir names if CTF data).
    subjects : list of str
        Subject directory names. These are sub-directories in outdir.
    ftype: None or str
        Extension of the preprocessed fif files. Default option is `_preproc-raw`.
    outdir : str
        Output directory.
    logsdir : str
        Directory to save log files to.
    reportdir : str
        Directory to save report files to.
    gen_report : bool
        Should we generate a report?
    overwrite : bool
        Should we overwrite the output file if it exists?
    skip_save: list or None (default)
        List of keys to skip writing to disk. If None, we don't skip any keys.    
    extra_funcs : list
        User-defined functions.
    covs : dict or pd.DataFrame
        Covariates to use for building the GLM design
    random_seed : 'auto' (default), int or None
        Random seed to set. If 'auto', a random seed will be generated. Random seeds are set for both Python and NumPy.
        If None, no random seed is set.
    verbose : str
        Level of info to print.
        Can be: ``'CRITICAL'``, ``'ERROR'``, ``'WARNING'``, ``'INFO'``, ``'DEBUG'`` or ``'NOTSET'``.
    mneverbose : str
        Level of info from MNE to print.
        Can be: ``'CRITICAL'``, ``'ERROR'``, ``'WARNING'``, ``'INFO'``, ``'DEBUG'`` or ``'NOTSET'``.
    strictrun : bool
        Should we ask for confirmation of user inputs before starting?
    dask_client : bool
        Indicate whether to use a previously initialised :py:class:`dask.distributed.Client <distributed.Client>` instance. 

    Returns
    -------
    list of bool
        Flags indicating whether preprocessing was successful for each input file.
        
    Notes
    -----
    If you are using a :py:class:`dask.distributed.Client <distributed.Client>` instance, you must initialise it
    before calling this function. For example:
    
    >>> from dask.distributed import Client
    >>> client = Client(threads_per_worker=1, n_workers=4)
    """
   
    if outdir is None:
        # Use the current working directory
        outdir = os.getcwd()

    # Validate the parent outdir - later do so for each subdirectory
    tmpoutdir = validate_outdir(outdir.split('{')[0])
    logsdir = validate_outdir(logsdir or tmpoutdir / "logs")
    reportdir = validate_outdir(reportdir or tmpoutdir / "preproc_report")

    # Initialise Loggers
    mne.set_log_level(mneverbose)
    if strictrun and verbose not in ['INFO', 'DEBUG']:
        # override logger level if strictrun requested but user won't see any info...
        verobse = 'INFO'
    logfile = os.path.join(logsdir, 'batch_preproc.log')
    osl_logger.set_up(log_file=logfile, level=verbose, startup=False)

    logger.info('Starting osl-ephys Batch Processing')

    # Set random seed
    if random_seed == 'auto':
        random_seed = set_random_seed()
    elif random_seed is None:
        pass
    else:
        set_random_seed(random_seed)
    
    # Check through inputs and parameters
    infiles, good_files_outnames, good_files = process_file_inputs(files)

    # Specify filenames for the output data
    if subjects is None:
        subjects = good_files_outnames
    else:
        if len(subjects) != len(good_files_outnames):
            logger.critical(
                f"Number of subjects ({len(subjects)}) does not match "
                f"number of good files {len(good_files_outnames)}. "
                "Please fix the subjects list or pass subjects=None."
            )

    if strictrun and click.confirm('Is this correct set of inputs?') is False:
        logger.critical('Stopping : User indicated incorrect number of input files')
        sys.exit(1)
    else:
        if strictrun:
            logger.info('User confirms input files')

    logger.info('Outputs saving to: {0}'.format(outdir))
    if strictrun and click.confirm('Is this correct output directory?') is False:
        logger.critical('Stopping : User indicated incorrect output directory')
        sys.exit(1)
    else:
        if strictrun:
            logger.info('User confirms output directory')

    config = load_config(config)
    config_str = pprint.PrettyPrinter().pformat(config)
    logger.info('Running config\n {0}'.format(config_str))
    if strictrun and click.confirm('Is this the correct config?') is False:
        logger.critical('Stopping : User indicated incorrect preproc config')
        sys.exit(1)
    else:
        if strictrun:
            logger.info('User confirms input config')

    if config['preproc'] is not None:
        # Create partial function with fixed options
        pool_func = partial(
            run_proc_chain,
            outdir=outdir,
            ftype=ftype,
            logsdir=logsdir,
            reportdir=reportdir,
            ret_dataset=False,
            gen_report=gen_report,
            overwrite=overwrite,
            skip_save=skip_save,
            extra_funcs=extra_funcs,
            random_seed=random_seed,
        )

        # Loop through input files to generate arguments for run_proc_chain
        args = []
        for infile, subject in zip(infiles, subjects):
            args.append((config, infile, subject))

        # Actually run the processes
        if dask_client:
            proc_flags = dask_parallel_bag(pool_func, args)
        else:
            proc_flags = [pool_func(*aa) for aa in args]
        
        if isinstance(proc_flags[0], tuple):
            group_inputs = [flag[1] for flag in proc_flags]
            proc_flags = [flag[0] for flag in proc_flags]
        
        osl_logger.set_up(log_file=logfile, level=verbose, startup=False)
        logger.info("Processed {0}/{1} files successfully".format( np.sum(proc_flags), len(proc_flags)))
        
        # Generate a report
        if gen_report and len(infiles) > 0:
            from ..report import preproc_report # avoids circular import
            preproc_report.gen_html_page(reportdir)
            
    else:
        group_inputs = [{"raw": infile} for infile in infiles]
        proc_flags = [os.path.exists(sub) for sub in infiles]
        
        osl_logger.set_up(log_file=logfile, level=verbose, startup=False)
        logger.info("No preprocessing steps specified. Skipping preprocessing.") 


    # start group processing
    if config['group'] is not None:
        logger.info("Starting Group Processing")
        logger.info(
            "Valid input files {0}/{1}".format(
                np.sum(proc_flags), len(proc_flags)
            )
        )
        dataset = {}
        skip_save=[]
        for key in group_inputs[0]:
            dataset[key] = [group_inputs[i][key] for i in range(len(group_inputs))]
            skip_save.append(key)
        
        if covs is not None:
            dataset['covs'] = covs
        dataset['fig'] = {}
        
        for stage in deepcopy(config["group"]):
            method, userargs = next(iter(stage.items()))
            # make sure the function always knows it's a group processing
            userargs['run_on_group'] = True
            target = userargs.get("target", "raw")  # Raw is default
            # skip.append(stage if userargs.get("skip_save") is True else None) # skip saving this stage to disk
            func = find_func(method, target=target, extra_funcs=extra_funcs)
                
            # Actual function call
            dataset = func(dataset, userargs)
        outbase = os.path.join(outdir, "{ftype}.{fext}")
        outnames = write_dataset(dataset, outbase, '', ftype='', overwrite=overwrite, skip=skip_save)

    # rerun the summary report
    if gen_report:
        from ..report import preproc_report # avoids circular import
        if preproc_report.gen_html_summary(reportdir, logsdir):
            logger.info("******************************" + "*" * len(str(reportdir)))
            logger.info(f"* REMEMBER TO CHECK REPORT: {reportdir} *")
            logger.info("******************************" + "*" * len(str(reportdir)))

    # Return flags
    return proc_flags


# ----------------------------------------------------------
# Main CLI user function


def main(argv=None):
    """Main function for command line interface.
    
    Parameters
    ----------
    argv : list
        Command line arguments.
    """    
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="Batch preprocess some fif files.")
    parser.add_argument("config", type=str, help="yaml defining preproc")
    parser.add_argument(
        "files",
        type=str,
        help="plain text file containing full paths to files to be processed",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Path to output directory to save data in",
    )
    parser.add_argument(
        "--logsdir", type=str, default=None, help="Path to logs directory"
    )
    parser.add_argument(
        "--reportdir", type=str, default=None, help="Path to report directory"
    )
    parser.add_argument(
        "--gen_report", type=bool, default=True, help="Should we generate a report?"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite previous output files if they're in the way",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="INFO",
        help="Set the logging level for osl-ephys functions",
    )
    parser.add_argument(
        "--mneverbose",
        type=str,
        default="WARNING",
        help="Set the logging level for MNE functions",
    )
    parser.add_argument(
        "--strictrun",
        action="store_true",
        help="Will ask the user for confirmation before starting",
    )

    parser.usage = parser.format_help()
    args = parser.parse_args(argv)

    run_proc_batch(**vars(args))


if __name__ == "__main__":
    main()
