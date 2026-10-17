"""Loading and checking a pipeline config.

A config is a YAML file, a YAML string or a dict with one list of steps per
section, in the order the sections run::

    preproc:
      - filter: {l_freq: 1, h_freq: 45}
      - bad_segments: {picks: mag}
    coreg:
      - extract_surfaces: {include_nose: false}
      - coregister: {use_nose: false}
    source_recon:
      - forward_model: {gridstep: 8}
      - beamform: {chantypes: [mag, grad]}
    parcellation:
      - parcellate: {parcellation_file: atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz}
    group:
      - fix_sign_ambiguity: {}

Any section can be left out. A ``meta`` section holds ``event_codes`` and
version checks.
"""

from __future__ import annotations


from packaging.version import Version
import re
import operator
from importlib.metadata import version
import logging
import yaml

from osl_ephys.pipeline import registry

logger = logging.getLogger(__name__)

SECTION_ORDER = list(registry.SECTIONS)


def load_config(config) -> dict:
    """Load a config and put it in the canonical form.

    Parameters
    ----------
    config : str or dict
        Path to a YAML file, a YAML string, or a dict.

    Returns
    -------
    config : dict
        One list of ``{step_name: kwargs}`` per section that is present,
        kwargs always a dict, plus ``meta``.
    """
    if not isinstance(config, (str, dict)):
        raise TypeError(f"config must be a str or dict, got {type(config)}.")

    if isinstance(config, str):
        try:
            with open(config, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except (UnicodeDecodeError, FileNotFoundError, OSError):
            config = yaml.load(config, Loader=yaml.FullLoader)
    config = dict(config)

    meta = config.pop("meta", None) or {}
    meta.setdefault("event_codes", None)
    meta.setdefault("versions", None)

    unknown = [key for key in config if key not in SECTION_ORDER]
    if unknown:
        raise KeyError(
            f"Unknown config section(s) {unknown}; sections are {SECTION_ORDER}."
        )

    sections = {}
    for section in SECTION_ORDER:
        steps = config.get(section)
        if steps in (None, "None", "null"):
            continue
        sections[section] = [_normalise_step(section, s) for s in steps]

    if not sections:
        raise KeyError(f"The config has no steps; sections are {SECTION_ORDER}.")

    for name, _ in sections.get("preproc", []):
        if name == "find_events" and meta["event_codes"] is None:
            raise KeyError(
                "event_codes must be given in meta if we are finding events."
            )

    return {
        **{
            s: [{n: kw} for n, kw in sections[s]]
            for s in SECTION_ORDER
            if s in sections
        },
        "meta": meta,
    }


def _normalise_step(section: str, stage) -> tuple[str, dict]:
    """A step as ``(name, kwargs)`` with kwargs a dict."""
    if isinstance(stage, str):
        return stage, {}
    if not isinstance(stage, dict) or len(stage) != 1:
        raise ValueError(
            f"Step '{stage}' in section '{section}' must be a dict with a single key."
        )
    ((name, kwargs),) = stage.items()
    if kwargs in (None, "None", "null"):
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ValueError(
            f"The options of step '{name}' must be a dict, got {type(kwargs)}."
        )
    return name, dict(kwargs)


def steps(config: dict, section: str) -> list[tuple[str, dict]]:
    """The ``(name, kwargs)`` steps of a section, empty if the section is absent."""
    return [tuple(s.items())[0] for s in config.get(section) or []]


def validate(
    config: dict, extra_funcs: list | None = None, columns: set | None = None
) -> None:
    """Check every step exists and the manifest has what the steps need.

    Parameters
    ----------
    config : dict
        A loaded config.
    extra_funcs : list of callable, optional
        Custom functions passed by the user.
    columns : set of str, optional
        Columns of the session manifest (reserved for checks that need them).

    Raises
    ------
    LookupError
        For a step nothing provides.
    """
    for section in SECTION_ORDER:
        for name, kwargs in steps(config, section):
            target = kwargs.get("target", "raw") if section == "preproc" else "raw"
            registry.find_function(name, section, extra_funcs, target=target)


def _parse_condition(cond):
    """Parse strings defining conditional statements.

    Borrowed from EMD package
    """
    name = re.split(r"[=<>!]", cond)[0]
    comp = cond[len(name) :]

    if comp[:2] == "==":
        func = operator.eq
    elif comp[:2] == "!=":
        func = operator.ne
    elif comp[:2] == "<=":
        func = operator.le
    elif comp[:2] == ">=":
        func = operator.ge
    elif comp[0] == "<":
        func = operator.lt
    elif comp[0] == ">":
        func = operator.gt
    else:
        print("Comparator not recognised!")

    val = comp.lstrip("!=<>")

    return (name, func, str(val))


def check_version(test_statement, mode="warn"):
    """Check whether the version of a package meets a specified condition.

    Parameters
    ----------
    test_statement : str
        Package version comparison string in the standard format expected by python installs.
        eg 'osl-ephys<1.0.0' or 'osl-ephys==0.6.dev0'
    mode : {'warn', 'assert'}
        Flag indicating whether to warn the user or raise an error if the comparison fails

    """
    test_module, comparator, target_version = _parse_condition(test_statement)

    test_version = Version(version(test_module))
    target_version = Version(target_version)

    if comparator(test_version, target_version) is False:
        msg = "Package '{}' version ({}) fails specified requirement ({})"
        msg = msg.format(test_module, test_version, test_statement)

        if mode == "warn":
            logger.warning(msg)
        elif mode == "assert":
            logger.warning(msg)
            raise AssertionError(msg)


def check_versions(config: dict) -> None:
    """Apply the ``version_assert`` and ``version_warn`` checks of ``meta``."""
    meta = config.get("meta", {})
    for spec in meta.get("version_assert") or []:
        check_version(spec, mode="assert")
    for spec in meta.get("version_warn") or []:
        check_version(spec, mode="warn")
