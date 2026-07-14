# ``ensure_dir`` / ``proc_userargs`` are shared with osl-ephys core and the
# manual_ica subpackage, so they live once in osl_ephys.utils. Re-exported here
# for the semp wrappers' ``from ..utils import proc_userargs`` imports.
from osl_ephys.utils import ensure_dir, proc_userargs  # noqa: F401


#: The implicit dataset "schema" the semp wrappers read. These keys are not on
#: the ``raw`` object -- they are seeded by the project ``initialize``
#: extra_func (the copy-paste template in the semp EEG-fMRI tutorial) or
#: produced by an upstream stage. Documented here so the contract is
#: discoverable in one place.
DATASET_SCHEMA = {
    # --- seeded by initialize ---
    'pf': "osl_pathfinder.Pathfinder for id<->path<->field lookups",
    'subject': "EEG recording id (from pf.path2id) -- names ckpt/output folders",
    'target_pth': "Path root for ckpt/ output (ckpt_report, summary, apply_ica)",
    'tr_interval': "fMRI TR in s (create_TR_epoch window, crop_TR; slice_reject uses 1/TR as its sideband width around each slice harmonic)",
    'slice_interval': "fMRI slice-timing interval in s (slice_reject harmonics)",
    'tr_event_key': "annotation label(s) of the volume trigger (crop_TR, create_TR_epoch)",
    'he_event_key': "annotation label(s) of the helium-pump trigger (create_He_epoch)",
    # --- produced by an upstream stage ---
    'tracer': "EEGTracer (init_tracer) or a dict of metric partials",
    'ica': "fitted mne ICA from ica_raw (slice_reject, apply_ica)",
    # epoch keys ('tr_ep', 'he_ep', 'sim_ep', ...) are produced by the
    # create_*_epoch / simulate_epoch wrappers and read by epoch_aas / epoch_obs.
}


def require_keys(dataset, keys, stage):
    """Assert the running ``dataset`` carries the keys a wrapper stage needs.

    The semp wrappers depend on the implicit :data:`DATASET_SCHEMA` -- keys the
    project ``initialize`` extra_func seeds, or that an upstream stage produces.
    A missing one otherwise surfaces as a bare ``KeyError`` deep inside the
    stage; this raises a single, actionable message up front instead. A key
    present but set to ``None`` counts as missing (matches ``dataset.get(...)
    is None`` guards elsewhere).

    Args:
        dataset (dict): the running dataset.
        keys (str | iterable[str]): the required key name(s).
        stage (str): the wrapper name, for the error message.

    Raises:
        KeyError: naming every missing key and the stage that needs it.
    """
    if isinstance(keys, str):
        keys = [keys]
    missing = [k for k in keys if dataset.get(k) is None]
    if missing:
        hints = "; ".join(
            f"dataset[{k!r}]" + (f" ({DATASET_SCHEMA[k]})" if k in DATASET_SCHEMA else "")
            for k in missing)
        raise KeyError(
            f"{stage} needs {hints} -- set it in the project 'initialize' "
            f"extra_func (see the template in the semp EEG-fMRI tutorial), or "
            f"run the upstream stage that produces it.")
