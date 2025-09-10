"""Wrappers for MNE functions to perform preprocessing.

    We have run_mne_anonymous which tries to run a method directly on a target
    object (typically an mne Raw or Epochs object).

    In addition, there are a set of wrapper functions for MNE methods which need
    a bit more option processing than the default - for example, converting input
    strings into arrays of frequencies

    Wrapper functions have priority and will be run rather than the direct method
    call if a wrapper is present. If no wrapper is present then we fall back to
    the direct method call.
    
    Most wrapper functions run on the `` object in `dataset` by default
    and the function docstrings assume this - but is most cases `mne.io.Raw` can
    be replaced with `mne.Epochs` (or `dataset['raw']` by `dataset['epochs']` and 
    the function will still work, e.g. :py:meth:`mne.Epochs.pick <mne.Epochs.pick>`.
    In order to apply the method to an object different from `mne.Raw`, the `target`
    argument can be specified in `userargs`. For example, `target: 'epochs'` can be
    specified in the userargs to apply the method to `dataset['epochs']` instead of
    `dataset['raw']`.
"""

# Authors: Andrew Quinn <a.quinn@bham.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>

import mne
import numpy as np

# Housekeeping for logging
import logging

logger = logging.getLogger(__name__)

# --------------------------------------------------------------
# MNE Raw/Epochs Object Methods
#


def run_mne_anonymous(dataset, userargs, method):
    """OSL-Batch function which runs a method directly on a target MNE object in ``dataset``,
    typically an :py:class:`mne.io.Raw <mne.io.Raw>` or :py:class:`mne.Epochs <mne.Epochs>` object.
    
    OSL Batch will first look for OSL/MNE wrapper functions for the method, and 
    otherwise will try to run the method directly on the target object.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Contains user arguments for the function.
    method: str
        See :py:class:`mne.io.Raw <mne.io.Raw>` and :py:class:`mne.Epochs <mne.Epochs>` for the available methods.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
        
    """    
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, method))
    logger.info("userargs: {0}".format(str(userargs)))
    if hasattr(dataset[target], method) and callable(getattr(dataset[target], method)):
        getattr(dataset[target], method)(**userargs)
    else:
        raise ValueError("Method '{0}' not found on target '{1}'".format(method, target))
    return dataset


# --------------------------------------------------------------
# Exceptions for which we can't just naively pass args through

# General wrapper functions - work on Raw and Epochs


def run_mne_notch_filter(dataset, userargs):
    """OSL-Batch wrapper for :py:meth:`mne.io.Raw.notch_filter <mne.io.Raw.notch_filter>`.

    This function calls :py:meth:`notch_filter <mne.io.Raw.notch_filter>` on
    an MNE object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictionary.

    Parameters
    ----------
    dataset : dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs : dict
        Dictionary of additional arguments to be passed to :py:meth:`mne.io.Raw.notch_filter <mne.io.Raw.notch_filter>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, "notch_filter"))
    logger.info("userargs: {0}".format(str(userargs)))
    freqs = userargs.pop("freqs")
    freqs = [
        float(freqs)
        if np.logical_or(type(freqs) == int, type(freqs) == float)
        else np.array(freqs.split(" ")).astype(float)
    ]
    dataset[target].notch_filter(freqs, **userargs)
    return dataset


def run_mne_pick(dataset, userargs):
    """OSL-Batch wrapper for :py:meth:`mne.io.Raw.pick <mne.io.Raw.pick>`.

    This function calls :py:meth:`pick <mne.io.Raw.pick>` on an MNE object in 
    ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.

    Parameters
    ----------
    dataset : dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs : dict
        Dictionary of additional arguments to be passed to :py:meth:`mne.io.Raw.pick <mne.io.Raw.pick>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.

    Notes
    -----
    In MNE-Batch, an example call would look like

    >>> preproc:
    >>>  - pick: {picks: 'meg'}

    By default, the :py:meth:`mne.io.Raw.pick <mne.io.Raw.pick>` will be
    called on ``dataset['raw']``, you can specify another options by specifying
    ``target`` in ``userargs``. For example:

    >>> preproc:
    >>>  - pick: {picks: 'meg', target: 'epochs'}

    Then the function or method will be called on ``dataset['epochs']`` instead.

    """
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, "pick"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset[target].pick(**userargs)
    return dataset


def run_mne_pick_channels(dataset, userargs):
    """OSL-Batch wrapper for :py:meth:`mne.io.Raw.pick_channels <mne.io.Raw.pick_channels>`.

    This function calls :py:meth:`pick_channels <mne.io.Raw.pick_channels>` on
    an MNE object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:meth:`mne.io.Raw.pick_channels <mne.io.Raw.pick_channels>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, "pick_channels"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset[target].pick_channels(**userargs)
    return dataset


def run_mne_pick_types(dataset, userargs):
    """OSL-Batch wrapper for :py:meth:`raw.pick_types <mne.io.Raw.pick_types>`.

    This function calls :py:meth:`pick_types <mne.io.Raw.pick_types>` on an MNE object in
    ``dataset``. Additional arguments on the MNE function can be specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:meth:`mne.io.Raw.pick_types <mne.io.Raw.pick_types>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, "pick_types"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset[target].pick_types(**userargs)
    return dataset


def run_mne_resample(dataset, userargs):
    """OSL-Batch wrapper for :py:meth:`mne.io.Raw.resample <mne.io.Raw.resample>`.

    This function calls :py:meth:`resample <mne.io.Raw.resample>` on
    an MNE object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:meth:`mne.io.Raw.resample <mne.io.Raw.resample>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, "resample"))
    logger.info("userargs: {0}".format(str(userargs)))
    if ("events" in dataset) and (dataset["events"] is not None):
        dataset[target], dataset["events"] = dataset[target].resample(
            events=dataset["events"], **userargs
        )
    else:
        dataset[target].resample(**userargs)
    return dataset


def run_mne_set_channel_types(dataset, userargs):
    """OSL-Batch wrapper for :py:meth:`mne.io.Raw.set_channel_types <mne.io.Raw.set_channel_types>`.

    This function calls :py:meth:`set_channel_types <mne.io.Raw.set_channel_types>` on
    an MNE object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:meth:`mne.io.Raw.set_channel_types <mne.io.Raw.set_channel_types>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, "set_channel_types"))
    logger.info("userargs: {0}".format(str(userargs)))
    # Separate function as we don't explode userargs
    dataset[target].set_channel_types(userargs)
    return dataset


def run_mne_interpolate_bads(dataset, userargs):
    """OSL-Batch wrapper for :py:meth:`mne.io.Raw.interpolate_bads <mne.io.Raw.interpolate_bads>`.

    This function calls :py:meth:`interpolate_bads <mne.io.Raw.interpolate_bads>` on
    an MNE object in ``dataset``. Importantly, it sets ``reset_bads`` to False by default.
    Additional arguments on the MNE function can be specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:meth:`mne.io.Raw.set_channel_types <mne.io.Raw.set_channel_types>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}.{1}".format(target, "interpolate_bads"))
    logger.info("userargs: {0}".format(str(userargs)))
    
    # remember the bad channels 
    original_bads = dataset[target].info['bads'].copy()
    dataset[target].interpolate_bads(**userargs)
    
    # if the bad channels were reset, we want to keep a local copy for the report
    if dataset[target].info['bads'] != original_bads:
        dataset[target].info['temp'] = {'bads': original_bads}
    return dataset


# Epochs Functions


def run_mne_drop_bad(dataset, userargs):
    """OSL-Batch wrapper for :py:meth:`mne.Epochs.drop_bad <mne.Epochs.drop_bad>`.

    This function calls :py:meth:`drop_bad <mne.Epochs.drop_bad>` on
    an MNE :py:class:`Epochs <mne.Epochs>` object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.
    
    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw`` and ``epochs``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:meth:`mne.Epochs.drop_bad <mne.Epochs.drop_bad>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "epochs")  # should only be epochs
    logger.info("MNE Stage - {0}.{1}".format(target, "drop_bad"))
    logger.info("userargs: {0}".format(str(userargs)))

    # Need to sanitise values in 'reject' dictionary - these are strings after
    # being read in from yaml.
    if "reject" in userargs:
        for key, value in userargs["reject"].items():
            userargs["reject"][key] = float(value)

    dataset[target] = dataset[target].drop_bad(**userargs)

    return dataset


# TFR


def run_mne_apply_baseline(dataset, userargs):
    """OSL-Batch wrapper for :py:meth:`epochs.apply_baseline <mne.Epochs.apply_baseline>`.

    This function calls :py:meth:`mne.Epochs.apply_baseline <mne.Epochs.apply_baseline>` on
    an MNE :py:class:`Epochs <mne.Epochs>` object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.
    
    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the keys ``raw`` and ``epochs``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:meth:`mne.Epochs.apply_baseline <mne.Epochs.apply_baseline>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "epochs")  # should only be epochs
    logger.info("MNE Stage - {0}.{1}".format(target, "apply_baseline"))
    logger.info("userargs: {0}".format(str(userargs)))

    freqs = np.array(userargs.pop("baseline").split(" ")).astype(float)
    dataset[target].apply_baseline(freqs, **userargs)
    return dataset


# --------------------------------------------------------------
# MNE Events and Epochs Object Methods
#

# These wrappers use MNE functions which use one or more different data types
# and typically create a new data object in the dataset dictionary.


def run_mne_find_events(dataset, userargs):
    """OSL-Batch wrapper for :py:func:`mne.find_events <mne.find_events>`.

    This function calls :py:func:`find_events <mne.find_events>` on
    an MNE :py:class:`Raw <mne.io.Raw>` object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:func:`mne.find_events <mne.find_events>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    logger.info("MNE Stage - {0}.{1}".format("mne", "find_events"))
    logger.info("userargs: {0}".format(str(userargs)))

    dataset["events"] = mne.find_events(dataset["raw"], **userargs)
    return dataset


def run_mne_epochs(dataset, userargs):
    """OSL-Batch wrapper for :py:class:`mne.Epochs <mne.Epochs>`.

    This function calls :py:class:`mne.Epochs <mne.Epochs>` on the ``raw``, ``events``, and ``event-id``
    keys in ``dataset``. Additional arguments on the MNE function can be specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the keys ``raw``, ``events``, and ``event-id``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:class:`mne.Epochs <mne.Epochs>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    logger.info("MNE Stage - {0}.{1}".format("mne", "epochs"))
    logger.info("userargs: {0}".format(str(userargs)))
    tmin = userargs.pop("tmin", -0.2)
    tmax = userargs.pop("tmax", 0.5)
    dataset["epochs"] = mne.Epochs(
        dataset["raw"], dataset["events"], dataset["event_id"], tmin, tmax, **userargs
    )
    return dataset


# --------------------------------------------------------------
# mne.preprocessing functions
def run_mne_annotate_amplitude(dataset, userargs):
    """OSL-Batch wrapper for :py:func:`mne.preprocessing.annotate_amplitude <mne.preprocessing.annotate_amplitude>`.

    This function calls :py:func:`annotate_amplitude <mne.preprocessing.annotate_amplitude>` on
    an MNE object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:func:`mne.preprocessing.annotate_amplitude <mne.preprocessing.annotate_amplitude>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    logger.info("MNE Stage - {0}.{1}".format("mne.preprocessing", "annotate_amplitude"))
    logger.info("userargs: {0}".format(str(userargs)))
    target = userargs.pop("target", "raw")

    # Import func - otherwise line is too long even for me
    from mne.preprocessing import annotate_amplitude

    bad_annotations, bad_channels = annotate_amplitude(dataset[target], **userargs)

    dataset[target].info["bads"].extend(bad_channels)

    # Can't combine annotations with different orig_times, the following line
    # fails if uncommented >>
    # dataset[target].set_annotations(dataset[target].annotations + bad_annotations)

    # ...so I'm extracting and reforming the muscle annotations here. Feels like
    # the line above would be better if it worked - must be missing something?
    onsets = []
    durations = []
    descriptions = []
    for ann in bad_annotations:
        onsets.append(ann["onset"])
        durations.append(ann["duration"])
        descriptions.append(ann["description"])

    dataset[target].annotations.append(onsets, durations, descriptions)

    return dataset


def run_mne_annotate_muscle_zscore(dataset, userargs):
    """OSL-Batch wrapper for :py:func:`mne.preprocessing.annotate_muscle_zscore <mne.preprocessing.annotate_muscle_zscore>`.

    This function calls :py:func:`annotate_muscle_zscore <mne.preprocessing.annotate_muscle_zscore>` on
    an MNE object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.
    
    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:func:`mne.preprocessing.annotate_muscle_zscore <mne.preprocessing.annotate_muscle_zscore>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    logger.info(
        "MNE Stage - {0}.{1}".format("mne.preprocessing", "annotate_muscle_zscore")
    )
    logger.info("userargs: {0}".format(str(userargs)))
    target = userargs.pop("target", "raw")

    # Import func - otherwise line is too long even for me
    from mne.preprocessing import annotate_muscle_zscore

    bad_annotations, _ = annotate_muscle_zscore(dataset[target], **userargs)

    # Can't combine annotations with different orig_times, the following line
    # fails if uncommented >>
    # dataset[target].set_annotations(dataset[target].annotations + bad_annotations)

    # ...so I'm extracting and reforming the muscle annotations here. Feels like
    # the line above would be better if it worked.
    onsets = []
    durations = []
    descriptions = []
    for ann in bad_annotations:
        onsets.append(ann["onset"])
        durations.append(ann["duration"])
        descriptions.append(ann["description"])

    dataset[target].annotations.append(onsets, durations, descriptions)

    return dataset


def run_mne_find_bad_channels_maxwell(dataset, userargs):
    """OSL-Batch wrapper for :py:func:`mne.preprocessing.find_bad_channels_maxwell <mne.preprocessing.find_bad_channels_maxwell>`.

    This function calls :py:func:`find_bad_channels_maxwell <mne.preprocessing.find_bad_channels_maxwell>` on
    an MNE :py:class:`Raw <mne.io.Raw>` object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:func:`mne.preprocessing.find_bad_channels_maxwell <mne.preprocessing.find_bad_channels_maxwell>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    logger.info(
        "MNE Stage - {0}.{1}".format("mne.preprocessing", "find_bad_channels_maxwell")
    )
    logger.info("userargs: {0}".format(str(userargs)))
    target = userargs.pop("target", "raw")

    # Import func - otherwise line is too long even for me
    from mne.preprocessing import find_bad_channels_maxwell

    # Run maxfilter
    noisy, flats, scores = find_bad_channels_maxwell(dataset[target], **userargs)

    dataset[target].info["bads"].extend(noisy + flat)

    return dataset


def run_mne_maxwell_filter(dataset, userargs):
    """OSL-Batch wrapper for :py:func:`mne.preprocessing.maxwell_filter <mne.preprocessing.maxwell_filter>`.

    This function calls :py:func:`maxwell_filter <mne.preprocessing.maxwell_filter>` on
    an MNE :py:class:`Raw <mne.io.Raw>` object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:func:`mne.preprocessing.maxwell_filter <mne.preprocessing.maxwell_filter>`.
        
    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    logger.info("MNE Stage - {0}.{1}".format("mne.preprocessing", "maxwell_filter"))
    logger.info("userargs: {0}".format(str(userargs)))
    target = userargs.pop("target", "raw")

    # Import func - otherwise line is too long even for me
    from mne.preprocessing import maxwell_filter

    # Run maxfilter
    dataset[target] = maxwell_filter(dataset[target], **userargs)

    return dataset


def run_mne_compute_current_source_density(dataset, userargs):
    """OSL-Batch wrapper for :py:func:`mne.preprocessing.compute_current_source_density <mne.preprocessing.compute_current_source_density>`.

    This function calls :py:func:`compute_current_source_density <mne.preprocessing.compute_current_source_density>` on
    an MNE object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:func:`mne.preprocessing.compute_current_source_density <mne.preprocessing.compute_current_source_density>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    logger.info(
        "MNE Stage - {0}.{1}".format(
            "mne.preprocessing", "compute_current_source_density"
        )
    )
    logger.info("userargs: {0}".format(str(userargs)))
    target = userargs.pop("target", "raw")

    # Import func - otherwise line is too long even for me
    from mne.preprocessing import compute_current_source_density

    # Run Laplacian
    dataset[target] = compute_current_source_density(dataset[target], **userargs)

    return dataset


# Time-frequency transforms


def run_mne_tfr_multitaper(dataset, userargs):
    """OSL-Batch wrapper for :py:func:`mne.time_frequency.tfr_multitaper <mne.time_frequency.tfr_multitaper>`.

    This function calls :py:func:`tfr_multitaper <mne.time_frequency.tfr_multitaper>` on
    an MNE :py:class:`Epochs <mne.Epochs>` or :py:class:`Evoked <mne.Evoked>` object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the keys ``raw``, and ``evoked`` or ``epochs``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:func:`mne.time_frequency.tfr_multitaper <mne.time_frequency.tfr_multitaper>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "epochs")
    logger.info("MNE Stage - {0} on {1}".format("tfr_multitaper", target))
    logger.info("userargs: {0}".format(str(userargs)))
    from mne.time_frequency import tfr_multitaper

    freqs = np.array(userargs.pop("freqs").split(" ")).astype("float")
    freqs = np.linspace(freqs[0], freqs[1], int(freqs[2]))
    out = tfr_multitaper(dataset[target], freqs, **userargs)
    if "return_itc" in userargs and userargs["return_itc"]:
        dataset["power"], dataset["itc"] = out
    else:
        dataset["power"] = out

    return dataset


def run_mne_tfr_morlet(dataset, userargs):
    """OSL-Batch wrapper for :py:func:`mne.time_frequency.tfr_morlet <mne.time_frequency.tfr_morlet>`.

    This function calls :py:func:`tfr_morlet <mne.time_frequency.tfr_morlet>` on
    an MNE :py:class:`Epochs <mne.Epochs>` or :py:class:`Evoked <mne.Evoked>` object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the keys ``raw``, and ``evoked`` or ``epochs``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:func:`mne.time_frequency.tfr_morlet <mne.time_frequency.tfr_morlet>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "epochs")
    logger.info("MNE Stage - {0} on {1}".format("tfr_morlet", target))
    logger.info("userargs: {0}".format(str(userargs)))
    from mne.time_frequency import tfr_morlet

    freqs = np.array(userargs.pop("freqs").split(" ")).astype("float")
    reqs = np.linspace(freqs[0], freqs[1], int(freqs[2]))
    dataset["power"], dataset["itc"] = tfr_morlet(dataset[target], freqs, **userargs)

    return dataset


def run_mne_tfr_stockwell(dataset, userargs):
    """OSL-Batch wrapper for :py:func:`mne.time_frequency.tfr_stockwell <mne.time_frequency.tfr_stockwell>`.

    This function calls :py:func:`tfr_stockwell <mne.time_frequency.tfr_stockwell>` on
    an MNE :py:class:`Epochs <mne.Epochs>` or :py:class:`Evoked <mne.Evoked>` object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the keys ``raw``, and ``evoked`` or ``epochs``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:func:`mne.time_frequency.tfr_stockwell <mne.time_frequency.tfr_stockwell>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "epochs")
    logger.info("MNE Stage - {0} on {1}".format("tfr_stockwell", target))
    logger.info("userargs: {0}".format(str(userargs)))
    from mne.time_frequency import tfr_stockwell

    out = tfr_stockwell(dataset[target], **userargs)
    if "return_itc" in userargs and userargs["return_itc"]:
        dataset["power"], dataset["itc"] = out
    else:
        dataset["power"] = out

    return dataset


# --------------------------------------------------------------
# OHBA/MNE ICA Tools
#
# Currently assuming ICA on Raw data - no Epochs support yet


def run_mne_ica_raw(dataset, userargs):
    """OSL-Batch wrapper for :py:class:`mne.preprocessing.ICA <mne.preprocessing.ICA>`.

    This function creates class :py:class:`ICA <mne.preprocessing.ICA>` 
    and fits it to an MNE object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary. The ``raw`` object in ``dataset`` is filtered (1 Hz high pass) before
    fitting the ICA.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:class:`mne.preprocessing.ICA <mne.preprocessing.ICA>` , 
        :py:meth:`mne.preprocessing.ICA.fit <mne.preprocessing.ICA.fit>`, and :py:meth:`mne.io.Raw.filter <mne.io.Raw.filter>` .

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}".format("mne.preprocessing.ICA"))
    logger.info("userargs: {0}".format(str(userargs)))
    
    # MNE recommends applying a high pass filter at 1 Hz before calculating
    # the ICA (make this adjustable by user):
    # https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html#filtering-to-remove-slow-drifts
    l_freq = userargs.pop("l_freq", 1)
    h_freq = userargs.pop("h_freq", None)
    filt_raw = dataset["raw"].copy().filter(l_freq=l_freq, h_freq=h_freq)

    # NOTE: **userargs doesn't work because 'picks' is in there
    noise_cov = userargs.pop("noise_cov", None)
    random_state = userargs.pop("random_state", None)
    method = userargs.pop("method", "fastica")
    fit_params = userargs.pop("fit_params", None)
    max_iter = userargs.pop("max_iter", 'auto')
    allow_ref_meg = userargs.pop("allow_ref_meg", False)
    ica = mne.preprocessing.ICA(n_components=userargs["n_components"], noise_cov=noise_cov, random_state=random_state,
                               method=method, fit_params=fit_params, max_iter=max_iter, allow_ref_meg=allow_ref_meg)
    ica.fit(filt_raw, picks=userargs["picks"])
    dataset["ica"] = ica
    return dataset


def run_mne_ica_autoreject(dataset, userargs):
    """OSL-Batch wrapper for :py:meth:`mne.preprocessing.ICA.find_bads_ecg <mne.preprocessing.ICA.find_bads_ecg>` and :py:meth:`mne.preprocessing.ICA.find_bads_eog <mne.preprocessing.ICA.find_bads_eog>`.

    This function identifies IC's that are deemed to correspond to ECG or EOG artifacts, as found by
    :py:meth:`find_bads_ecg <mne.preprocessing.ICA.find_bads_ecg>` and 
    :py:meth:`find_bads_eog <mne.preprocessing.ICA.find_bads_eog>` on
    the ``raw`` and ``ica`` objects in ``dataset``. Additional arguments on the MNE functions can be
    specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:meth:`mne.preprocessing.ICA.find_bads_ecg <mne.preprocessing.ICA.find_bads_ecg>`
        and :py:meth:`mne.preprocessing.ICA.find_bads_eog <mne.preprocessing.ICA.find_bads_eog>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "raw")
    logger.info("OSL Stage - {0}".format("ICA Autoreject"))
    logger.info("userargs: {0}".format(str(userargs)))

    # User specified arguments and their defaults
    eogmeasure = userargs.pop("eogmeasure", "correlation")
    eogthreshold = userargs.pop("eogthreshold", 0.35)
    eogmethod = userargs.pop("eogmethod", "default")
    ecgmethod = userargs.pop("ecgmethod", "ctps")
    ecgthreshold = userargs.pop("ecgthreshold", "auto")
    remove_components = userargs.pop("apply", True)
    
    # Reject components based on the EOG channel
    if not (eogmethod is None or eogmethod == "None"):
        eog_indices, eog_scores = dataset["ica"].find_bads_eog(
            dataset["raw"], threshold=eogthreshold, measure=eogmeasure,
        )
        dataset["ica"].exclude.extend(eog_indices)
        logger.info("Marking {0} as EOG ICs".format(len(eog_indices)))

    # Reject components based on the ECG channel
    if not (ecgmethod is None or ecgmethod == "None"):
        ecg_indices, ecg_scores = dataset["ica"].find_bads_ecg(
            dataset["raw"], threshold=ecgthreshold, method=ecgmethod
        )
        dataset["ica"].exclude.extend(ecg_indices)
        logger.info("Marking {0} as ECG ICs".format(len(ecg_indices)))

    # Remove the components from the data if requested
    if remove_components:
        logger.info("Removing selected components from raw data")
        dataset["ica"].apply(dataset["raw"])
    else:
        logger.info("Components were not removed from raw data")

    return dataset


def run_mne_apply_ica(dataset, userargs):
    """OSL-Batch wrapper for :py:meth:`mne.preprocessing.ICA.apply <mne.preprocessing.ICA.apply>`.

    This function creates class :py:meth:`mne.preprocessing.ICA.apply <mne.preprocessing.ICA.apply>` 
    and fits it to an MNE object in ``dataset``. Additional arguments on the MNE function can be
    specified as a dictonary.

    Parameters
    ----------
    dataset: dict
        Dictionary containing at least an MNE object with the key ``raw``.
    userargs: dict
        Dictionary of additional arguments to be passed to :py:meth:`mne.preprocessing.ICA.apply <mne.preprocessing.ICA.apply>`.

    Returns
    -------
    dataset: dict
        Input dictionary containing MNE objects that have been modified in place.
    """    
    target = userargs.pop("target", "raw")
    logger.info("MNE Stage - {0}".format("ica.apply"))
    logger.info("userargs: {0}".format(str(userargs)))
    dataset["raw"] = dataset["ica"].apply(dataset["raw"], **userargs)
    return dataset
