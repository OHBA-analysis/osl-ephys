"""A session's files: reading raw data, and reading and writing the dataset the
preprocessing produces (the data, its events, epochs and ICA).

A dataset is a dict with the keys ``raw``, ``events``, ``event_id``,
``epochs`` and ``ica``, plus whatever a step adds. It is written as
``<id>_preproc-raw.fif`` and, when present, ``<id>_events.npy``,
``<id>_event-id.yml``, ``<id>_epo.fif``, ``<id>_ica.fif`` (``_ica-2.fif``,
... for further ICA runs) and a pickle for any other key.
"""

from __future__ import annotations

import inspect
import logging
import os
import pickle
import re
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import yaml
import matplotlib.pyplot as plt

logger = logging.getLogger("osl_ephys")

_READERS = {
    ".fif": mne.io.read_raw_fif,
    ".edf": mne.io.read_raw_edf,
    ".ds": mne.io.read_raw_ctf,
    ".vhdr": mne.io.read_raw_brainvision,
    ".set": mne.io.read_raw_eeglab,
    ".con": mne.io.read_raw_kit,
    ".sqd": mne.io.read_raw_kit,
    ".bdf": mne.io.read_raw_bdf,
    ".mff": mne.io.read_raw_egi,
}
_CURRY = (".dat", ".dap", ".rs3", ".cdt", ".cdt.dpa", ".cdt.cef", ".cef")


def import_data(infile, preload=True):
    """Read a raw recording, picking the reader from the file name.

    Parameters
    ----------
    infile : str
        The file: fif, CTF (.ds or .meg4), BTI (c,rfDC), edf, BrainVision
        (.vhdr), EEGLAB (.set), KIT (.con, .sqd), bdf, EGI (.mff) or Curry;
        anything else is given to :func:`mne.io.read_raw`.
    preload : bool, optional
        Load the data into memory.

    Returns
    -------
    raw : mne.io.Raw
        The recording.
    """
    if not isinstance(infile, str):
        raise ValueError(f"infile must be a str, got {type(infile)}.")
    if " " in infile:
        raise ValueError("filename cannot contain spaces.")
    logger.info(f"Importing {infile}")
    directory, name = os.path.split(infile)
    ext = os.path.splitext(infile)[1]
    if name == "c,rfDC":
        has_headshape = os.path.isfile(os.path.join(directory, "hs_file"))
        return mne.io.read_raw_bti(
            infile,
            head_shape_fname="hs_file" if has_headshape else None,
            preload=preload,
        )
    if ext == ".meg4":
        return mne.io.read_raw_ctf(directory, preload=preload)
    if ext.lower() == ".edf":
        return mne.io.read_raw_edf(infile, preload=preload)
    if ext in _READERS:
        return _READERS[ext](infile, preload=preload)
    if ext in _CURRY:
        return mne.io.read_raw_curry(infile, preload=preload)
    try:
        return mne.io.read_raw(infile, preload=preload)
    except Exception:
        raise ValueError(f"Unable to determine the file type of {infile}")


def get_config_from_fif(inst):
    """The configs recorded in a preprocessed file's ``info["description"]``.

    Returns
    -------
    configs : list of dict
        One config per preprocessing run, oldest first.
    """
    from osl_ephys.pipeline.config import load_config

    texts = re.findall(
        "%% config start %%(.*?)%% config end %%",
        inst.info["description"] or "",
        flags=re.DOTALL,
    )
    return [load_config(text) for text in texts]


def append_preproc_info(dataset, config, extra_funcs=None):
    """Record the config (and the source of any custom functions) in
    ``info["description"]`` of the data and the epochs, with the date and
    the osl-ephys version."""
    from osl_ephys import __version__

    text = (
        "\n\nOSL-EPHYS BATCH PROCESSING APPLIED ON "
        f"{datetime.today().strftime('%d/%m/%Y %H:%M:%S')} \n"
        f"VERSION: {__version__}\n"
        f"%% config start %% \n{config} \n%% config end %%"
    )
    for func in extra_funcs or []:
        text += (
            "\n\nCUSTOM FUNCTIONS USED:\n%% extra_funcs start %% \n"
            f"{inspect.getsource(func)}\n%% extra_funcs end %%"
        )
    for key in ("raw", "epochs"):
        if dataset.get(key) is not None:
            info = dataset[key].info
            info["description"] = (info["description"] or "") + text
    return dataset


def write_dataset(
    dataset, outbase, run_id, ftype="preproc-raw", overwrite=False, skip=None
):
    """Write a dataset's keys to files, one per key.

    Parameters
    ----------
    dataset : dict
        The dataset.
    outbase : str
        A template with ``{run_id}``, ``{ftype}`` and ``{fext}``, e.g.
        ``"outdir/{run_id}_{ftype}.{fext}"``.
    run_id : str
        The session id.
    ftype : str, optional
        The suffix of the data file.
    overwrite : bool, optional
        Overwrite existing files.
    skip : list of str, optional
        Keys not to write.

    Returns
    -------
    outnames : dict
        The file written for each key.
    """
    skip = skip or []
    for string in ("_preproc-raw", "_raw"):
        run_id = run_id.replace(string, "")

    def name(ftype, fext):
        return outbase.format(run_id=run_id, ftype=ftype, fext=fext)

    outnames = {"raw": None}
    if "raw" not in skip:
        outnames["raw"] = name(ftype, "fif")
        if Path(outnames["raw"]).exists() and not overwrite:
            raise ValueError(
                f"{outnames['raw']} exists; pass overwrite=True to replace it."
            )
        logger.info(f"Saving dataset['raw'] as {outnames['raw']}")
        dataset["raw"].save(outnames["raw"], overwrite=overwrite)

    def wanted(key):
        return key in dataset and key not in skip and dataset[key] is not None

    if wanted("events"):
        outnames["events"] = name("events", "npy")
        np.save(outnames["events"], dataset["events"])
    if wanted("event_id"):
        outnames["event_id"] = name("event-id", "yml")
        with open(outnames["event_id"], "w") as f:
            yaml.dump(dataset["event_id"], f)
    if wanted("epochs"):
        outnames["epochs"] = name("epo", "fif")
        dataset["epochs"].save(outnames["epochs"], overwrite=overwrite)
    if wanted("ica"):
        icas = dataset["ica"] if isinstance(dataset["ica"], list) else [dataset["ica"]]
        outnames["ica"] = []
        for n, ica in enumerate(icas, start=1):
            outnames["ica"].append(name("ica" if n == 1 else f"ica-{n}", "fif"))
            ica.save(outnames["ica"][-1], overwrite=overwrite)
        if len(outnames["ica"]) == 1:
            outnames["ica"] = outnames["ica"][0]
    if wanted("tfr"):
        outnames["tfr"] = name("tfr", "fif")
        dataset["tfr"].save(outnames["tfr"], overwrite=overwrite)
    if wanted("glm"):
        outnames["glm"] = name("glm", "pkl")
        dataset["glm"].save_pkl(outnames["glm"], overwrite=overwrite)
    if wanted("fig"):
        outnames["fig"] = {}
        for key, fig in dataset["fig"].items():
            outnames["fig"][key] = name(key, "png")
            fig.savefig(outnames["fig"][key])
    for key in dataset:
        if key in outnames or key in skip or dataset[key] is None:
            continue
        outnames[key] = name(key, "pkl")
        if not os.path.exists(outnames[key]) or overwrite:
            with open(outnames[key], "wb") as f:
                pickle.dump(dataset[key], f)
    for key, value in outnames.items():
        if value is not None and key != "raw":
            logger.info(f"Saved dataset['{key}'] as {value}")
    return outnames


def read_dataset(fif, preload=False, ftype=None):
    """Read a preprocessed file and the files saved alongside it.

    Parameters
    ----------
    fif : str
        The data file.
    preload : bool, optional
        Load the data into memory.
    ftype : str, optional
        The suffix of the data file (``preproc-raw`` when the name has it,
        else the part after the last underscore).

    Returns
    -------
    dataset : dict
        ``raw``, ``events``, ``event_id``, ``epochs`` and ``ica`` (a list
        when the session has several ICA runs); missing ones are None.
    """
    fif = str(fif)
    logger.info(f"Reading {fif}")
    raw = mne.io.read_raw_fif(fif, preload=preload)
    if ftype is None:
        if "preproc-raw" in fif:
            ftype = "preproc-raw"
        elif len(fif.split("_")) < 2:
            raise ValueError(f"Unable to guess the file type of {fif}; pass ftype.")
        else:
            ftype = fif.split("_")[-1].split(".")[-2]
    ftype = ftype + ".fif"

    def sidecar(suffix):
        path = Path(fif.replace(ftype, suffix))
        return path if path.exists() else None

    dataset = {
        "raw": raw,
        "events": None,
        "event_id": None,
        "epochs": None,
        "ica": None,
    }
    if (path := sidecar("events.npy")) is not None:
        dataset["events"] = np.load(path)
    if (path := sidecar("event-id.yml")) is not None:
        with open(path) as f:
            dataset["event_id"] = yaml.load(f, Loader=yaml.Loader)
    if (path := sidecar("epo.fif")) is not None:
        dataset["epochs"] = mne.read_epochs(path)
    icas = [sidecar("ica.fif")]
    n = 2
    while (path := sidecar(f"ica-{n}.fif")) is not None:
        icas.append(path)
        n += 1
    icas = [mne.preprocessing.read_ica(p) for p in icas if p is not None]
    if icas:
        dataset["ica"] = icas[0] if len(icas) == 1 else icas
    return dataset


def plot_preproc_flowchart(
    config,
    outname=None,
    show=False,
    stagecol="wheat",
    startcol="red",
    fig=None,
    ax=None,
    title=None,
):
    """A flowchart of the ``preproc`` and ``group`` steps of a config.

    Parameters
    ----------
    config : str or dict
        The config.
    outname : str, optional
        File to save to.
    show : bool, optional
        Show the figure.
    stagecol, startcol : str, optional
        Colours of the step and section boxes.
    fig, ax : optional
        Figure and axes to draw on.
    title : str, optional
        Title.

    Returns
    -------
    fig, ax
    """
    from osl_ephys.pipeline.config import load_config

    config = load_config(config)
    if ax is None or fig is None:
        fig = plt.figure(figsize=(8, 12))
        plt.subplots_adjust(top=0.95, bottom=0.05)
        ax = plt.subplot(111, frame_on=False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or "osl-ephys processing recipe", fontsize=24)

    stages = [{"input": ""}]
    for section in ("preproc", "group"):
        if config.get(section):
            stages += [{section: ""}, *config[section]]
    stages.append({"output": ""})
    stage_height = 1 / len(stages)
    box = dict(boxstyle="round", facecolor=stagecol, alpha=1, pad=0.3)
    startbox = dict(boxstyle="round", facecolor=startcol, alpha=1)
    font = {"family": "serif", "color": "k", "weight": "normal", "size": 16}
    ax.arrow(
        0.5,
        1,
        0.0,
        -1 + 0.02,
        fc="k",
        ec="k",
        head_width=0.045,
        head_length=0.035,
        length_includes_head=True,
    )
    for i, stage in enumerate(stages):
        method, userargs = next(iter(stage.items()))
        method = method.replace("_", r"\_")
        is_section = method in ("input", "preproc", "group", "output")
        ax.text(
            0.5,
            1 - stage_height * i,
            f"$\\bf{{{method}{'' if is_section else ':'}}}$ {str(userargs)[1:-1]}",
            ha="center",
            va="center",
            bbox=startbox if is_section else box,
            fontdict=font,
            wrap=True,
        )
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.25, 0.75)
    if outname is not None:
        fig.savefig(outname, dpi=300, transparent=True)
    if show:
        fig.show()
    return fig, ax
