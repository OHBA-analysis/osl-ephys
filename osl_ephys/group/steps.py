"""Sign flipping as config steps: ``fix_sign_ambiguity`` flips one session to
a template (a parcellation step) and ``sign_flip`` picks the template and
flips every session (the group step)."""

from __future__ import annotations

import logging
from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt

from osl_ephys.group import sign_flipping
from osl_ephys.pipeline.registry import step

logger = logging.getLogger("osl_ephys")


def _parc_file(dataset: dict, id_: str) -> Path:
    """The parcellated file of a session, in the same output directory."""
    outdir = dataset["paths"]["session_dir"].parent
    return outdir / id_ / f"{id_}_parc-raw.fif"


def _template_cov(parc_file, n_embeddings, standardize):
    raw = mne.io.read_raw_fif(str(parc_file), preload=True, verbose=False)
    return sign_flipping.calc_cov(
        raw.get_data(picks="misc").T, n_embeddings, standardize
    )


_SOLVERS = {
    "spectral": sign_flipping.find_flips_spectral,
    "belief_propagation": sign_flipping.find_flips_belief_propagation,
    "sdp": sign_flipping.find_flips_sdp,
}


def _flip_file(
    parc_file, template_cov, out, n_embeddings, standardize, method, options
):
    """Flip the parcel signs of one file to match the template covariance.

    ``method`` is ``coordinate_ascent`` (the default search) or one of the
    solvers in ``_SOLVERS``, with that solver's options. Returns the
    correlation with the template before and after flipping.
    """
    if method not in (None, "coordinate_ascent", "random", *_SOLVERS):
        raise ValueError(
            f"method must be 'coordinate_ascent' or one of {sorted(_SOLVERS)}."
        )
    raw = mne.io.read_raw_fif(str(parc_file), preload=True, verbose=False)
    picks = mne.pick_types(raw.info, misc=True)
    cov = sign_flipping.calc_cov(raw.get_data(picks).T, n_embeddings, standardize)
    before = sign_flipping.calc_corr(cov, template_cov, n_embeddings)
    if method in _SOLVERS:
        flips, metrics = _SOLVERS[method](cov, template_cov, n_embeddings, **options)
        after = metrics[1]
    else:
        flips, after = sign_flipping.find_flips(
            cov, template_cov, n_channels=len(picks), n_embeddings=n_embeddings
        )
    logger.info(f"Template correlation {before:.3f} -> {after:.3f} ({method})")
    raw._data[picks] *= np.asarray(flips)[:, None]
    raw.save(str(out), overwrite=True, verbose=False)
    return float(before), float(after)


@step("parcellation")
def fix_sign_ambiguity(dataset, userargs):
    """Flip the parcel signs of this session to match a template session.

    Options: ``template`` (the id of the template session), ``n_embeddings``,
    ``standardize``, ``method`` (``coordinate_ascent``, ``spectral``,
    ``belief_propagation`` or ``sdp``, plus that solver's options). Writes
    ``<id>_sflip_parc-raw.fif``.
    """
    n_embeddings = userargs.pop("n_embeddings", 15)
    standardize = userargs.pop("standardize", True)
    method = userargs.pop("method", "coordinate_ascent")
    template = userargs.pop("template")
    cov = _template_cov(_parc_file(dataset, template), n_embeddings, standardize)
    id_ = dataset["session"]["id"]
    out = _parc_file(dataset, id_).with_name(f"{id_}_sflip_parc-raw.fif")
    _flip_file(
        _parc_file(dataset, id_), cov, out, n_embeddings, standardize, method, userargs
    )
    return dataset


@step("group")
def sign_flip(dataset, userargs):
    """Pick the template session and flip every session's parcel signs to it.

    The template is the session whose embedded covariance correlates best
    with the others' on average (or ``template`` if given). Options:
    ``n_embeddings``, ``standardize``, ``template``, ``method`` (as for
    ``fix_sign_ambiguity``).
    """
    userargs.pop("run_on_group", None)
    n_embeddings = userargs.pop("n_embeddings", 15)
    standardize = userargs.pop("standardize", True)
    method = userargs.pop("method", "coordinate_ascent")
    template = userargs.pop("template", None)
    sessions = dataset["sessions"]
    outdir = Path(dataset["outdir"])
    parc_files = {
        s["id"]: outdir / s["id"] / f"{s['id']}_parc-raw.fif" for s in sessions
    }
    covs = {
        id_: _template_cov(f, n_embeddings, standardize)
        for id_, f in parc_files.items()
        if f.exists()
    }
    if template is None:
        scores = {
            a: np.mean(
                [
                    sign_flipping.calc_corr(covs[a], covs[b], n_embeddings)
                    for b in covs
                    if b != a
                ]
                or [0]
            )
            for a in covs
        }
        template = max(scores, key=scores.get)
        logger.info(f"Template session: {template}")
    dataset["template"] = template
    results = {}
    for id_, f in parc_files.items():
        if id_ in covs:
            out = f.with_name(f"{id_}_sflip_parc-raw.fif")
            before, after = _flip_file(
                f, covs[template], out, n_embeddings, standardize, method, userargs
            )
            results[id_] = {"before": before, "after": after}
    if dataset.get("qc_dir") is not None:
        _save_qc(dataset["qc_dir"], template, results)
    return dataset


def _save_qc(qc_dir, template, results):
    """The template and each session's correlation with it, and their plot."""
    from osl_ephys.report import headless, save_figure, save_summary

    qc_dir = Path(qc_dir) / "sign_flip"
    save_summary(qc_dir, {"template": template, **results})
    ids = list(results)
    with headless():
        _plot(qc_dir, ids, results, save_figure)


def _plot(qc_dir, ids, results, save_figure):
    fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(ids)), 4))
    x = np.arange(len(ids))
    ax.bar(x - 0.2, [results[i]["before"] for i in ids], 0.4, label="Before flipping")
    ax.bar(x + 0.2, [results[i]["after"] for i in ids], 0.4, label="After flipping")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=90, fontsize=8)
    ax.set_ylabel("Correlation with the template covariance")
    ax.legend(frameon=False)
    save_figure(fig, qc_dir / "sign_flip.webp")
