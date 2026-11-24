"""Sign flipping as config steps: ``fix_sign_ambiguity`` flips one session to
a template (a parcellation step) and ``sign_flip`` picks the template and
flips every session (the group step)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import mne
import numpy as np
import matplotlib.pyplot as plt

from osl_ephys.group import sign_flipping
from osl_ephys.pipeline.registry import step

logger = logging.getLogger("osl_ephys")


def _parc_files(outdir, ids, estimator=None):
    """The source estimator and the parcellated file of each session that has
    one, ``<id>_<estimator>-parc-raw.fif``. ``estimator`` picks one when the
    sessions were parcellated with several."""
    found = {}
    for id_ in ids:
        pattern = re.compile(re.escape(id_) + r"_([^-]+)-parc-raw\.fif")
        files = (Path(outdir) / id_).glob("*-parc-raw.fif")
        matches = [(pattern.fullmatch(f.name), f) for f in files]
        found[id_] = {match[1]: f for match, f in matches if match}
    estimators = sorted({name for files in found.values() for name in files})
    if estimator is None:
        if len(estimators) > 1:
            raise ValueError(
                f"The sessions were parcellated with {', '.join(estimators)}; "
                "choose one with the estimator option."
            )
        estimator = estimators[0] if estimators else None
    estimator = None if estimator is None else str(estimator).lower()
    return estimator, {
        id_: files[estimator] for id_, files in found.items() if estimator in files
    }


def _read_parcels(parc_file):
    """The parcellated raw and the indices of its parcel (misc) channels."""
    raw = mne.io.read_raw_fif(str(parc_file), preload=True, verbose=False)
    return raw, mne.pick_types(raw.info, misc=True)


def _parcel_data(raw, picks):
    """The parcel time courses used for the covariance, (n_samples, n_parcels).

    Segments annotated as bad are left out, as osl-signflip does, so they do not
    affect the flips; the flips are then applied to the whole recording.
    """
    return raw.get_data(picks, reject_by_annotation="omit", verbose=False).T


def _cov(parc_file, n_embeddings, standardize):
    """The time-delay embedded covariance of a parcellated file."""
    raw, picks = _read_parcels(parc_file)
    return sign_flipping.compute_covariance(
        _parcel_data(raw, picks), n_embeddings, standardize
    )


def _sign_flip_options(userargs):
    """Split the step options into the common ones and the method's.

    Returns ``(n_embeddings, standardize, method, method_kwargs)``, where
    ``method_kwargs`` are the hyperparameters of ``method``. The method and its options are
    checked here so a typo fails before any file is read.
    """
    n_embeddings = userargs.pop("n_embeddings", 15)
    standardize = userargs.pop("standardize", True)
    method = userargs.pop("method", sign_flipping.DEFAULT_METHOD)
    valid = sign_flipping.get_method_hyperparameters(method)
    unknown = sorted(set(userargs) - set(valid))
    if unknown:
        raise ValueError(
            f"Unknown option(s) {', '.join(map(repr, unknown))} for sign flipping "
            f"method {method!r}. Valid options are: n_embeddings, standardize, "
            f"method, template, estimator, {', '.join(valid)}."
        )
    return n_embeddings, standardize, method, dict(userargs)


def _flip_file(
    parc_file, template_cov, out, n_embeddings, standardize, method, method_kwargs
):
    """Flip the parcel signs of one file to match the template covariance.

    Returns the correlation with the template before and after flipping.
    """
    raw, picks = _read_parcels(parc_file)
    cov = sign_flipping.compute_covariance(
        _parcel_data(raw, picks), n_embeddings, standardize
    )
    before = sign_flipping.covariance_matrix_correlation(
        cov, template_cov, diag_offset=n_embeddings
    )
    flips, after = sign_flipping.find_flips(
        cov, template_cov, n_embeddings, method, **method_kwargs
    )
    logger.info(f"Template correlation {before:.3f} -> {after:.3f} ({method})")
    raw._data[picks] *= np.asarray(flips)[:, None]
    raw.save(str(out), overwrite=True, verbose=False)
    return float(before), float(after)


def _copy_template(parc_file, out, template_cov, n_embeddings):
    """Save the template session unflipped (it is not aligned to itself)."""
    raw, _ = _read_parcels(parc_file)
    raw.save(str(out), overwrite=True, verbose=False)
    corr = sign_flipping.covariance_matrix_correlation(
        template_cov, template_cov, diag_offset=n_embeddings
    )
    return float(corr), float(corr)


@step("parcellation")
def fix_sign_ambiguity(dataset, userargs):
    """Flip the parcel signs of this session to match a template session.

    Options: ``template`` (the id of the template session), ``n_embeddings``
    (default 15, must be odd), ``standardize`` (default true), ``method`` (one
    of ``sdp`` (default), ``spectral``, ``belief_propagation`` or ``random``;
    the methods come from osl-signflip), ``estimator`` (the source estimator,
    when several were parcellated), and the hyperparameters of the method, e.g.
    ``{method: sdp, sdp_n_restarts: 10}`` or
    ``{method: random, n_restarts: 3, n_iter: 2500, max_flips: 20}``
    (see :func:`osl_ephys.group.sign_flipping.get_method_hyperparameters`). Reads
    ``<id>_<estimator>-parc-raw.fif`` and writes
    ``<id>_sflip-<estimator>-parc-raw.fif``.
    """
    template = userargs.pop("template")
    estimator = userargs.pop("estimator", None)
    n_embeddings, standardize, method, method_kwargs = _sign_flip_options(userargs)
    id_ = dataset["session"]["id"]
    estimator, parc_files = _parc_files(
        dataset["paths"]["session_dir"].parent, [template, id_], estimator
    )
    cov = _cov(parc_files[template], n_embeddings, standardize)
    out = parc_files[id_].with_name(f"{id_}_sflip-{estimator}-parc-raw.fif")
    if id_ == template:
        _copy_template(parc_files[id_], out, cov, n_embeddings)
    else:
        _flip_file(
            parc_files[id_], cov, out, n_embeddings, standardize, method, method_kwargs
        )
    return dataset


@step("group")
def sign_flip(dataset, userargs):
    """Pick the template session and flip every session's parcel signs to it.

    The template is the median session in terms of the similarity of the
    sign-invariant (absolute-valued) covariances, as chosen by
    ``osl_signflip.find_template`` (or ``template`` if given). The template
    session is saved unflipped. Options: as for ``fix_sign_ambiguity``, with
    ``template`` optional.
    """
    userargs.pop("run_on_group", None)
    template = userargs.pop("template", None)
    estimator = userargs.pop("estimator", None)
    n_embeddings, standardize, method, method_kwargs = _sign_flip_options(userargs)
    sessions = dataset["sessions"]
    outdir = Path(dataset["outdir"])
    estimator, parc_files = _parc_files(outdir, [s["id"] for s in sessions], estimator)
    covs = {id_: _cov(f, n_embeddings, standardize) for id_, f in parc_files.items()}
    if template is None:
        ids = list(covs)
        index = sign_flipping.find_template(
            np.stack([covs[id_] for id_ in ids]),
            diag_offset=n_embeddings,
            use_tqdm=False,
        )
        template = ids[index]
        logger.info(f"Template session: {template}")
    elif template not in covs:
        raise ValueError(f"Template session {template!r} has no parcellated file.")
    dataset["template"] = template
    results = {}
    for id_, f in parc_files.items():
        out = f.with_name(f"{id_}_sflip-{estimator}-parc-raw.fif")
        if id_ == template:
            before, after = _copy_template(f, out, covs[template], n_embeddings)
        else:
            before, after = _flip_file(
                f, covs[template], out, n_embeddings, standardize, method, method_kwargs
            )
        results[id_] = {"before": before, "after": after}
    if dataset.get("qc_dir") is not None:
        _save_qc(dataset["qc_dir"], template, method, method_kwargs, results)
    return dataset


def _save_qc(qc_dir, template, method, method_kwargs, results):
    """The template, the method, and each session's correlation with the
    template, and their plot."""
    from osl_ephys.report import headless, save_figure, save_summary

    qc_dir = Path(qc_dir) / "sign_flip"
    save_summary(
        qc_dir,
        {
            "template": template,
            "method": method,
            "method_kwargs": method_kwargs,
            **results,
        },
    )
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
