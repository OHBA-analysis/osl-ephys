"""The step registry: which function runs for a step name in a config section.

A step is registered with :func:`step` (or :func:`register_module` for a
module of wrappers) under a section and a name. The config refers to steps by
name without the ``run_mne_``/``run_osl_`` prefix, as it always has; the
full function name is accepted too.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from functools import partial
from typing import Callable

import mne

SECTIONS = ("maxfilter", "preproc", "coreg", "source_recon", "parcellation", "group")

# Prefixes stripped from a wrapper's name to get its config name
PREFIXES = ("run_mne_", "run_osl_", "run_")


@dataclass(frozen=True)
class Step:
    """A registered step.

    Parameters
    ----------
    func : callable
        The function that runs the step.
    section : str
        The config section it belongs to.
    name : str
        The name the config uses.
    per : {"session", "subject"}
        Whether it runs once per session or once per subject (e.g. surface
        extraction, shared by the sessions of a subject).
    """

    func: Callable
    section: str
    name: str
    per: str = "session"


_REGISTRY: dict[str, dict[str, Step]] = {section: {} for section in SECTIONS}

# The modules that register steps, imported on first lookup so that the
# subpackages can import each other freely
STEP_MODULES = (
    "osl_ephys.maxfilter.steps",
    "osl_ephys.preproc.steps",
    "osl_ephys.rhino.steps",
    "osl_ephys.source_recon.steps",
    "osl_ephys.parcellation.steps",
    "osl_ephys.group.steps",
)
_loaded = False


def load_steps() -> None:
    """Import every module that registers steps (once)."""
    global _loaded
    if not _loaded:
        _loaded = True
        for module in STEP_MODULES:
            importlib.import_module(module)


def config_name(func_name: str) -> str:
    """The name a config uses for a wrapper, e.g. ``filter`` for ``run_mne_filter``."""
    for prefix in PREFIXES:
        if func_name.startswith(prefix):
            return func_name[len(prefix) :]
    return func_name


def register(
    func: Callable,
    section: str,
    name: str | None = None,
    per: str = "session",
) -> Step:
    """Register a function as a step; see :class:`Step` for the arguments."""
    if section not in SECTIONS:
        raise ValueError(f"section must be one of {SECTIONS}, got '{section}'.")
    if per not in ("session", "subject"):
        raise ValueError(f"per must be 'session' or 'subject', got '{per}'.")
    name = name or config_name(func.__name__)
    entry = Step(func, section, name, per)
    _REGISTRY[section][name] = entry
    if func.__name__ != name:
        _REGISTRY[section][func.__name__] = entry
    return entry


def step(section: str, name: str | None = None, per: str = "session"):
    """Decorator form of :func:`register`."""

    def decorate(func):
        register(func, section, name, per)
        return func

    return decorate


def register_module(
    module, section: str, prefixes: tuple[str, ...] = ("run_mne_", "run_osl_"), **kwargs
) -> list[Step]:
    """Register every function of a module whose name starts with one of the prefixes."""
    steps = []
    for func_name, func in inspect.getmembers(module, inspect.isfunction):
        if func_name.startswith(prefixes) and func.__module__ == module.__name__:
            steps.append(register(func, section, **kwargs))
    return steps


def registered(section: str | None = None) -> dict:
    """The registered steps of a section (or of every section), by config name."""
    load_steps()
    if section is None:
        return {s: registered(s) for s in SECTIONS}
    return {
        name: entry for name, entry in _REGISTRY[section].items() if entry.name == name
    }


def section_of(name: str) -> str | None:
    """The section a step name belongs to, or None if it is not registered."""
    load_steps()
    for section in SECTIONS:
        if name in _REGISTRY[section]:
            return section
    return None


_MNE_TARGETS = {
    "raw": mne.io.Raw,
    "epochs": mne.Epochs,
    "power": mne.time_frequency.EpochsTFR,
    "itc": mne.time_frequency.EpochsTFR,
}


def find_function(
    name: str, section: str, extra_funcs: list | None = None, target: str = "raw"
) -> Step:
    """The step to run for a name in a section.

    Looked up in this order: a custom function passed by the user (matched by
    its name), a registered step of the section, and for ``preproc`` a method
    of the MNE object named by ``target``.

    Parameters
    ----------
    name : str
        Step name from the config.
    section : str
        Config section.
    extra_funcs : list of callable, optional
        Custom functions passed by the user.
    target : str, optional
        MNE object a bare method name applies to: ``raw``, ``epochs``,
        ``power`` or ``itc``.

    Returns
    -------
    step : Step
        The step, with its calling convention.

    Raises
    ------
    LookupError
        If nothing provides the step.
    """
    load_steps()
    for func in extra_funcs or []:
        if func.__name__ == name or config_name(func.__name__) == name:
            return Step(func, section, name)

    if name in _REGISTRY[section]:
        return _REGISTRY[section][name]

    if section == "preproc":
        cls = _MNE_TARGETS.get(target)
        if cls is not None and callable(getattr(cls, name, None)):
            from osl_ephys.preproc.steps import mne_method

            return Step(partial(mne_method, method=name), section, name)

    elsewhere = section_of(name)
    hint = f" It is a '{elsewhere}' step." if elsewhere else ""
    raise LookupError(f"No step named '{name}' in section '{section}'.{hint}")
