"""The steps of the ``preproc`` section of a config.

Each step takes the dataset dict of a session (``raw``, ``events``,
``event_id``, ``epochs``, ``ica``, ...) and its options, and returns the
dataset. Most steps run on ``dataset["raw"]``; ``target: epochs`` runs one
on the epochs instead. A step named after a method of the target (``filter``,
``crop``, ...) that has no step here calls that method with the options
(:func:`mne_method`).
"""

from __future__ import annotations

import importlib
import logging

import mne
import numpy as np
from scipy import stats

from osl_ephys.pipeline.registry import register, step
from osl_ephys.preproc import functions

logger = logging.getLogger("osl_ephys")


def _target(dataset, userargs, default="raw"):
    target = userargs.pop("target", default)
    return target, dataset[target]


def _floats(value):
    """A number, or a space-separated string of numbers, as floats."""
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return np.array(value.split()).astype(float)
    return np.asarray(value, dtype=float)


def _latest_ica(dataset):
    """The ICA the ICA steps act on: the last one fitted."""
    ica = dataset.get("ica")
    if ica is None:
        raise ValueError("No ICA has been fitted; run ica_raw first.")
    return ica[-1] if isinstance(ica, list) else ica


def _add_ica(dataset, ica):
    """Keep every ICA fitted in the session, in order (they are saved as
    ``_ica.fif``, ``_ica-2.fif``, ...)."""
    if dataset.get("ica") is None:
        dataset["ica"] = ica
    elif isinstance(dataset["ica"], list):
        dataset["ica"].append(ica)
    else:
        dataset["ica"] = [dataset["ica"], ica]
    return dataset


def _warn_if_rectified(obj, what):
    """Filtering a rectified signal (a power or amplitude envelope, which is
    never negative) is usually a mistake, so say so."""
    try:
        data = obj.get_data(picks="data")
    except Exception:
        return
    if data.size and data.min() >= 0:
        logger.warning(
            f"The data is non-negative everywhere, as a rectified or power signal "
            f"is; {what} it is usually a mistake."
        )


def mne_method(dataset, userargs, method):
    """Call a method of the target MNE object with the options.

    The fallback for any step name that is a method of ``mne.io.Raw``,
    ``mne.Epochs`` or the time-frequency objects.
    """
    target, obj = _target(dataset, userargs)
    logger.info(f"{target}.{method}({userargs})")
    if not callable(getattr(obj, method, None)):
        raise ValueError(f"Method '{method}' not found on target '{target}'")
    if method in ("filter", "notch_filter", "savgol_filter"):
        _warn_if_rectified(obj, "filtering")
    getattr(obj, method)(**userargs)
    return dataset


@step("preproc")
def notch_filter(dataset, userargs):
    """MNE ``notch_filter``; ``freqs`` is a number or a space-separated list."""
    target, obj = _target(dataset, userargs)
    _warn_if_rectified(obj, "filtering")
    obj.notch_filter(_floats(userargs.pop("freqs")), **userargs)
    return dataset


@step("preproc")
def pick(dataset, userargs):
    """MNE ``pick``."""
    _, obj = _target(dataset, userargs)
    obj.pick(**userargs)
    return dataset


@step("preproc")
def pick_channels(dataset, userargs):
    """MNE ``pick_channels``."""
    _, obj = _target(dataset, userargs)
    obj.pick_channels(**userargs)
    return dataset


@step("preproc")
def pick_types(dataset, userargs):
    """MNE ``pick_types``."""
    _, obj = _target(dataset, userargs)
    obj.pick_types(**userargs)
    return dataset


@step("preproc")
def resample(dataset, userargs):
    """MNE ``resample``, resampling the events with the data."""
    target, obj = _target(dataset, userargs)
    if dataset.get("events") is not None:
        dataset[target], dataset["events"] = obj.resample(
            events=dataset["events"], **userargs
        )
    else:
        obj.resample(**userargs)
    return dataset


@step("preproc")
def set_channel_types(dataset, userargs):
    """MNE ``set_channel_types``; the options are the channel-to-type mapping."""
    _, obj = _target(dataset, userargs)
    obj.set_channel_types(userargs)
    return dataset


@step("preproc")
def interpolate_bads(dataset, userargs):
    """MNE ``interpolate_bads``. The interpolated channels stay in
    ``info["temp"]["bads"]`` for the report when ``reset_bads`` clears them."""
    _, obj = _target(dataset, userargs)
    bads = list(obj.info["bads"])
    obj.interpolate_bads(**userargs)
    if obj.info["bads"] != bads:
        obj.info["temp"] = {"bads": bads}
    return dataset


@step("preproc")
def drop_bad(dataset, userargs):
    """MNE ``Epochs.drop_bad``; the ``reject`` thresholds are read as floats."""
    target, obj = _target(dataset, userargs, "epochs")
    if "reject" in userargs:
        userargs["reject"] = {k: float(v) for k, v in userargs["reject"].items()}
    dataset[target] = obj.drop_bad(**userargs)
    return dataset


@step("preproc")
def apply_baseline(dataset, userargs):
    """MNE ``Epochs.apply_baseline``; ``baseline`` is two space-separated times."""
    _, obj = _target(dataset, userargs, "epochs")
    obj.apply_baseline(_floats(userargs.pop("baseline")), **userargs)
    return dataset


@step("preproc")
def find_events(dataset, userargs):
    """:func:`mne.find_events` on the data, into ``dataset["events"]``."""
    dataset["events"] = mne.find_events(dataset["raw"], **userargs)
    return dataset


@step("preproc")
def epochs(dataset, userargs):
    """:class:`mne.Epochs` from the data, the events and the config's
    ``event_codes``; ``tmin`` and ``tmax`` default to -0.2 and 0.5."""
    tmin = userargs.pop("tmin", -0.2)
    tmax = userargs.pop("tmax", 0.5)
    dataset["epochs"] = mne.Epochs(
        dataset["raw"], dataset["events"], dataset["event_id"], tmin, tmax, **userargs
    )
    return dataset


def _append_annotations(obj, annotations):
    """Add annotations to an object (whose annotations may have another
    orig_time, so they cannot be concatenated directly)."""
    obj.annotations.append(
        [a["onset"] for a in annotations],
        [a["duration"] for a in annotations],
        [a["description"] for a in annotations],
    )


@step("preproc")
def annotate_amplitude(dataset, userargs):
    """:func:`mne.preprocessing.annotate_amplitude`: the flat and peak-to-peak
    annotations are added and the flat channels marked bad."""
    _, obj = _target(dataset, userargs)
    annotations, bads = mne.preprocessing.annotate_amplitude(obj, **userargs)
    obj.info["bads"].extend(bads)
    _append_annotations(obj, annotations)
    return dataset


@step("preproc")
def annotate_muscle_zscore(dataset, userargs):
    """:func:`mne.preprocessing.annotate_muscle_zscore`."""
    _, obj = _target(dataset, userargs)
    annotations, _ = mne.preprocessing.annotate_muscle_zscore(obj, **userargs)
    _append_annotations(obj, annotations)
    return dataset


@step("preproc")
def find_bad_channels_maxwell(dataset, userargs):
    """:func:`mne.preprocessing.find_bad_channels_maxwell`: the noisy and flat
    channels are marked bad."""
    _, obj = _target(dataset, userargs)
    noisy, flat, _ = mne.preprocessing.find_bad_channels_maxwell(obj, **userargs)
    obj.info["bads"].extend(noisy + flat)
    return dataset


@step("preproc")
def maxwell_filter(dataset, userargs):
    """:func:`mne.preprocessing.maxwell_filter`."""
    target, obj = _target(dataset, userargs)
    dataset[target] = mne.preprocessing.maxwell_filter(obj, **userargs)
    return dataset


@step("preproc")
def compute_current_source_density(dataset, userargs):
    """:func:`mne.preprocessing.compute_current_source_density`."""
    target, obj = _target(dataset, userargs)
    dataset[target] = mne.preprocessing.compute_current_source_density(obj, **userargs)
    return dataset


def _tfr(dataset, userargs, function, needs_freqs=True):
    from mne import time_frequency

    target, obj = _target(dataset, userargs, "epochs")
    args = []
    if needs_freqs:
        # freqs: "start stop n", a linear spacing
        start, stop, n = _floats(userargs.pop("freqs"))
        args.append(np.linspace(start, stop, int(n)))
    out = getattr(time_frequency, function)(obj, *args, **userargs)
    if userargs.get("return_itc"):
        dataset["power"], dataset["itc"] = out
    else:
        dataset["power"] = out
    return dataset


@step("preproc")
def tfr_multitaper(dataset, userargs):
    """:func:`mne.time_frequency.tfr_multitaper` on the epochs; ``freqs`` is
    ``"start stop n"``. Writes ``power`` (and ``itc`` with ``return_itc``)."""
    return _tfr(dataset, userargs, "tfr_multitaper")


@step("preproc")
def tfr_morlet(dataset, userargs):
    """:func:`mne.time_frequency.tfr_morlet` on the epochs; see tfr_multitaper."""
    return _tfr(dataset, userargs, "tfr_morlet")


@step("preproc")
def tfr_stockwell(dataset, userargs):
    """:func:`mne.time_frequency.tfr_stockwell` on the epochs."""
    return _tfr(dataset, userargs, "tfr_stockwell", needs_freqs=False)


@step("preproc")
def ica_raw(dataset, userargs):
    """Fit :class:`mne.preprocessing.ICA` on a filtered copy of the data
    (``l_freq`` 1 Hz by default). Options: ``n_components``, ``picks``,
    ``method``, ``fit_params``, ``max_iter``, ``random_state``,
    ``noise_cov``, ``allow_ref_meg``, ``l_freq``, ``h_freq``. A second
    ``ica_raw`` in a config (e.g. for the EEG channels) is kept alongside
    the first.
    """
    userargs.pop("target", None)
    l_freq = userargs.pop("l_freq", 1)
    h_freq = userargs.pop("h_freq", None)
    picks = userargs.pop("picks")
    filtered = dataset["raw"].copy().filter(l_freq=l_freq, h_freq=h_freq)
    ica = mne.preprocessing.ICA(
        n_components=userargs.pop("n_components"),
        noise_cov=userargs.pop("noise_cov", None),
        random_state=userargs.pop("random_state", None),
        method=userargs.pop("method", "fastica"),
        fit_params=userargs.pop("fit_params", None),
        max_iter=userargs.pop("max_iter", "auto"),
        allow_ref_meg=userargs.pop("allow_ref_meg", False),
    )
    ica.fit(filtered, picks=picks)
    return _add_ica(dataset, ica)


@step("preproc")
def ica_autoreject(dataset, userargs):
    """Mark the components of the last ICA that MNE's ``find_bads_eog`` and
    ``find_bads_ecg`` flag, and remove them from the data (``apply: false``
    keeps them). Options: ``eogmeasure``, ``eogthreshold``, ``eogmethod``
    (None skips EOG), ``ecgmethod`` (None skips ECG), ``ecgthreshold``."""
    userargs.pop("target", None)
    ica = _latest_ica(dataset)
    eog_measure = userargs.pop("eogmeasure", "correlation")
    eog_threshold = userargs.pop("eogthreshold", 0.35)
    eog_method = userargs.pop("eogmethod", "default")
    ecg_method = userargs.pop("ecgmethod", "ctps")
    ecg_threshold = userargs.pop("ecgthreshold", "auto")
    apply = userargs.pop("apply", True)
    if eog_method not in (None, "None"):
        inds, _ = ica.find_bads_eog(
            dataset["raw"], threshold=eog_threshold, measure=eog_measure
        )
        ica.exclude.extend(inds)
        logger.info(f"{len(inds)} EOG components")
    if ecg_method not in (None, "None"):
        inds, _ = ica.find_bads_ecg(
            dataset["raw"], threshold=ecg_threshold, method=ecg_method
        )
        ica.exclude.extend(inds)
        logger.info(f"{len(inds)} ECG components")
    if apply:
        ica.apply(dataset["raw"])
    return dataset


@step("preproc")
def apply_ica(dataset, userargs):
    """Remove the excluded components of the last ICA from the data
    (:meth:`mne.preprocessing.ICA.apply`)."""
    userargs.pop("target", None)
    dataset["raw"] = _latest_ica(dataset).apply(dataset["raw"], **userargs)
    return dataset


@step("preproc")
def ica_label(dataset, userargs):
    """ICA with automatic labelling by mne-icalabel
    (:func:`~osl_ephys.preproc.functions.ica_label`); the artefact
    components are removed from the data."""
    userargs.pop("target", None)
    dataset["raw"], ica, _ = functions.ica_label(dataset["raw"], **userargs)
    return _add_ica(dataset, ica)


@step("preproc")
def ica_ecg_eog_correlation(dataset, userargs):
    """ICA with the artefact components found by ECG and EOG correlation
    (:func:`~osl_ephys.preproc.functions.ica_ecg_eog_correlation`); they are
    removed from the data."""
    userargs.pop("target", None)
    dataset["raw"], ica, _ = functions.ica_ecg_eog_correlation(
        dataset["raw"], **userargs
    )
    return _add_ica(dataset, ica)


@step("preproc")
def bad_segments(dataset, userargs):
    """Mark bad segments (:func:`~osl_ephys.preproc.functions.detect_bad_segments`)."""
    userargs.pop("target", None)
    dataset["raw"] = functions.detect_bad_segments(dataset["raw"], **userargs)
    return dataset


@step("preproc")
def bad_channels(dataset, userargs):
    """Mark bad channels (:func:`~osl_ephys.preproc.functions.detect_bad_channels`)."""
    userargs.pop("target", None)
    dataset["raw"] = functions.detect_bad_channels(dataset["raw"], **userargs)
    return dataset


@step("preproc")
def drop_bad_epochs(dataset, userargs):
    """Drop outlying epochs (:func:`~osl_ephys.preproc.functions.drop_bad_epochs`)."""
    userargs.pop("target", None)
    if dataset.get("epochs") is None:
        raise ValueError("No epochs to drop from; run the epochs step first.")
    dataset["epochs"] = functions.drop_bad_epochs(dataset["epochs"], **userargs)
    return dataset


@step("preproc")
def decimate_headshape_points(dataset, userargs):
    """Decimate a dense headshape scan
    (:func:`~osl_ephys.preproc.functions.decimate_headshape_points`)."""
    userargs.pop("target", None)
    dataset["raw"] = functions.decimate_headshape_points(dataset["raw"], **userargs)
    return dataset


@step("preproc")
def amm(dataset, userargs):
    """Adaptive multipole model interference rejection for OPM data
    (:func:`~osl_ephys.preproc.amm.apply_amm`): options ``li``, ``le``,
    ``window`` and ``corr_lim``."""
    from osl_ephys.preproc.amm import apply_amm

    userargs.pop("target", None)
    dataset["raw"], _ = apply_amm(dataset["raw"], **userargs)
    return dataset


@step("preproc")
def read_dataset(dataset, userargs):
    """Read the files saved alongside the data (events, epochs, ICA) into the
    dataset, and ``extra_keys`` (space-separated; ``.pkl`` unless a suffix is
    given) from the same directory."""
    import pickle
    from pathlib import Path

    import yaml

    from osl_ephys.preproc.dataset import read_dataset as read

    fif = str(dataset["raw"].filenames[0])
    ftype = userargs.pop("ftype", None)
    extra = userargs.pop("extra_keys", "")
    read_in = read(fif, preload=False, ftype=ftype)
    for key in ("events", "event_id", "epochs", "ica"):
        dataset[key] = read_in[key]
    if ftype is None:
        ftype = (
            "preproc-raw" if "preproc-raw" in fif else fif.split("_")[-1].split(".")[-2]
        )
    for key in extra.split():
        path = Path(fif.replace(f"{ftype}.fif", key))
        if "." not in path.name:
            path = path.with_suffix(".pkl")
        name = key.split(".")[0]
        if not path.exists():
            continue
        if path.suffix == ".pkl":
            with open(path, "rb") as f:
                dataset[name] = pickle.load(f)
        elif path.suffix == ".npy":
            dataset[name] = np.load(path)
        elif path.suffix == ".yml":
            with open(path) as f:
                dataset[name] = yaml.load(f, Loader=yaml.Loader)
    return dataset


@step("preproc")
def zscore_present_data(dataset, userargs):
    """Z-score the covariates named in ``keys`` (space-separated, optionally
    in brackets), with the missing values set to zero."""
    keys = userargs.pop("keys")
    if isinstance(keys, str):
        keys = keys.strip("[]").split()
    for key in keys:
        z = stats.zscore(dataset["covs"][key], nan_policy="omit")
        z[np.isnan(dataset["covs"][key])] = 0
        dataset["covs"][key] = z
    return dataset


@step("preproc")
def glm_add_regressor(dataset, userargs):
    """Add a regressor to the GLM design: ``rtype`` is ``Constant``,
    ``Categorical`` (``codes``, or ``unique`` for one regressor per value of
    the covariate ``key``), ``Parametric`` (``key``, ``preproc``) or
    ``MeanEffects`` (``key``)."""
    import glmtools

    if not isinstance(dataset.get("design_config"), glmtools.design.DesignConfig):
        dataset["design_config"] = glmtools.design.DesignConfig()
    design = dataset["design_config"]
    rtype = userargs.pop("rtype", None)
    name = userargs.pop("name", None)
    codes = userargs.pop("codes", None)
    preproc = userargs.pop("preproc", None)
    key = userargs.pop("key", None)
    if rtype == "Constant":
        design.add_regressor(name, rtype)
    elif rtype == "Categorical":
        if codes == "unique":
            for code in np.unique(dataset["covs"][key]):
                design.add_regressor(name=f"{name}_{code}", rtype=rtype, codes=code)
        else:
            codes = _floats(codes[0] if isinstance(codes, list) else codes)
            codes = codes[0] if len(codes) == 1 else codes
            design.add_regressor(name=name, rtype=rtype, codes=codes)
    elif rtype == "Parametric":
        design.add_regressor(name=name, rtype=rtype, datainfo=key, preproc=preproc)
    elif rtype == "MeanEffects":
        design.add_regressor(name=name + "_{0}", rtype=rtype, datainfo=key)
    else:
        raise ValueError(f"Unknown regressor type: {rtype}")
    return dataset


@step("preproc")
def glm_add_contrast(dataset, userargs):
    """Add a contrast to the GLM design: ``simple`` adds one per regressor;
    otherwise ``name`` and ``values`` (a mapping, or ``unique`` for the mean
    over the values of the covariate ``key``)."""
    design = dataset["design_config"]
    if userargs.pop("simple", False):
        design.add_simple_contrasts()
        return dataset
    name = userargs.pop("name", None)
    values = userargs.pop("values", None)
    key = userargs.pop("key", None)
    if values == "unique":
        unique = np.unique(dataset["covs"][key])
        values = {f"{key}_{v}": 1 / len(unique) for v in unique}
    else:
        values = {
            k: float(eval(v)) if isinstance(v, str) else float(v)
            for k, v in values.items()
        }
    design.add_contrast(name=name, values=values)
    return dataset


@step("preproc")
def glm_fit(dataset, userargs):
    """Fit a GLM: ``method`` is ``epochs`` or ``spectrum``, on the ``target``
    (the epochs or the data; the group GLMs on the fitted session GLMs) into
    ``dataset[name]``, with the design summary, efficiency and leverage
    plots as figures."""
    from osl_ephys.glm import (
        glm_epochs,
        glm_spectrum,
        group_glm_epochs,
        group_glm_spectrum,
    )

    on_group = userargs.pop("run_on_group", False)
    method = userargs.pop("method", None)
    if method is None:
        raise ValueError("method not specified")
    is_epochs = method in ("epochs", "glm_epochs")
    target = userargs.pop("target", None) or (
        "glm" if on_group else ("epochs" if is_epochs else "raw")
    )
    name = userargs.pop("name", None) or ("group_glm" if on_group else "glm")
    metric = userargs.pop("metric", "copes")
    plots = {
        p: userargs.pop(f"plot_{p}", True)
        for p in ("summary", "efficiency", "leverage")
    }
    baseline = userargs.pop("baseline", None)
    if baseline is not None:
        baseline = _floats(
            baseline.strip("[]") if isinstance(baseline, str) else baseline
        )
    if is_epochs:
        if on_group:
            dataset[name] = group_glm_epochs(
                dataset[target],
                dataset["design_config"],
                dataset["covs"],
                metric,
                baseline,
            )
        else:
            dataset[name] = glm_epochs(dataset["design_config"], dataset[target])
    else:
        if on_group:
            dataset[name] = group_glm_spectrum(
                dataset[target],
                dataset["design_config"],
                dataset["covs"],
                metric,
                baseline,
            )
        else:
            regressors = {}
            for kind in ("reg_categorical", "reg_ztrans", "reg_unitmax"):
                keys = userargs.pop(kind, None)
                if keys is None:
                    continue
                if isinstance(keys, str) and keys.startswith("["):
                    keys = keys.strip("[]").split()
                regressors[kind] = dataset["covs"][keys]
            dataset[name] = glm_spectrum(dataset[target], **regressors, **userargs)
    design = dataset[name].design
    for plot, wanted in plots.items():
        if wanted:
            dataset["fig"][f"{name}design_{plot}"] = getattr(design, f"plot_{plot}")(
                show=False
            )
    return dataset


@step("preproc")
def glm_permutations(dataset, userargs):
    """Permutation test of a fitted group GLM (``target``): ``method`` is
    ``epochs`` or ``spectrum``, ``type`` is ``max`` or ``cluster``, with the
    ``contrast`` (and first-level ``fl_contrast``) by name; the significant
    clusters at ``threshold`` are plotted into a figure."""
    import matplotlib.pyplot as plt

    from osl_ephys.glm import ClusterPermuteGLMSpectrum, MaxStatPermuteGLMSpectrum
    from osl_ephys.glm.glm_base import SensorClusterPerm, SensorMaxStatPerm

    userargs.pop("run_on_group", None)
    target = userargs.pop("target", "group_glm")
    name = userargs.pop("name", "group_glm_perm")
    method = userargs.pop("method", None)
    kind = userargs.pop("type", None)
    if method is None or kind is None:
        raise ValueError("method and type (max or cluster) must be given")
    threshold = userargs.pop("threshold", 95)
    plot_sig = userargs.pop("plot_sig", True)
    glm = dataset[target]
    contrast = glm.contrast_names.index(userargs.pop("contrast", None))
    fl_contrast = userargs.pop("fl_contrast", 0)
    if fl_contrast != 0:
        fl_contrast = glm.fl_contrast_names.index(fl_contrast)
    is_epochs = method in ("epochs", "glm_epochs")
    classes = {
        ("max", True): SensorMaxStatPerm,
        ("maxstat", True): SensorMaxStatPerm,
        ("max", False): MaxStatPermuteGLMSpectrum,
        ("maxstat", False): MaxStatPermuteGLMSpectrum,
        ("cluster", True): SensorClusterPerm,
        ("cluster", False): ClusterPermuteGLMSpectrum,
    }
    dataset[name] = classes[(kind, is_epochs)](glm, contrast, fl_contrast, **userargs)
    if plot_sig:
        fig, ax = plt.subplots()
        dataset[name].plot_sig_clusters(threshold, ax=ax)
        dataset["fig"][f"{name}sig{threshold}"] = fig
    return dataset


@step("preproc")
def manual_ica(dataset, userargs):
    """Fit an ICA and write the browser review page for labelling its
    components (:mod:`osl_ephys.preproc.ica.manual`); the ICA is not applied
    until ``apply_manual_ica`` or the ``osl-ephys ica apply`` command."""
    from osl_ephys.preproc.ica.manual import manual_ica as run

    _seed_manual_ica(dataset)
    return run(dataset, userargs)


@step("preproc")
def apply_manual_ica(dataset, userargs):
    """Apply a saved ICA excluding ``bad_ics``, the in-pipeline counterpart
    of the ``osl-ephys ica apply`` command."""
    from osl_ephys.preproc.ica.manual import apply_ica as run

    _seed_manual_ica(dataset)
    return run(dataset, userargs)


def _seed_manual_ica(dataset):
    """The manual ICA package names its files by ``subject`` under
    ``target_pth``: the session id and the run's output directory."""
    if dataset.get("subject") is None and "session" in dataset:
        dataset["subject"] = dataset["session"]["id"]
    if dataset.get("target_pth") is None and "paths" in dataset:
        dataset["target_pth"] = dataset["paths"]["session_dir"].parent


# The SEMP (simultaneous EEG-fMRI) steps, imported when one runs so that
# their heavy dependencies are not loaded otherwise
_SEMP_STEPS = (
    "voltage_correction",
    "cleanup",
    "mid_crop",
    "init_tracer",
    "summary",
    "ckpt_report",
    "crop_TR",
    "crop_by_epoch",
    "create_epoch",
    "create_TR_epoch",
    "create_He_epoch",
    "simulate_epoch",
    "epoch_ssp",
    "epoch_aas",
    "epoch_obs",
    "slice_reject",
    "start_timer",
    "end_timer",
)


def _semp_step(name):
    def run(dataset, userargs):
        wrappers = importlib.import_module("osl_ephys.preproc.semp.wrappers")
        return getattr(wrappers, name)(dataset, userargs)

    run.__name__ = name
    run.__doc__ = (
        f"The SEMP step ``{name}`` (:func:`osl_ephys.preproc.semp.wrappers.{name}`)."
    )
    return run


for _name in _SEMP_STEPS:
    register(_semp_step(_name), "preproc")

# The GLM steps also run in the group section, on the sessions' fitted GLMs
for _func in (glm_add_regressor, glm_add_contrast, glm_fit, glm_permutations):
    register(_func, "group")
