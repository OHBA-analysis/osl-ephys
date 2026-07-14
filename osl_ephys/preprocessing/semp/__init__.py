"""semp --- Simultaneous EEG-fMRI Preprocessing Toolbox (merged into osl-ephys).

Layout
------
- top level (this package) --- the preprocessing wrappers that slot into
  osl-ephys's ``run_proc_chain`` / ``run_proc_batch`` (``epoch_aas``,
  ``epoch_obs``, ``slice_reject`` ...), plus ``metric`` (EEGTracer / psd
  bands). (``garbage.py`` is dead scratch, never imported.)
- ``vis`` --- user-facing plotting you call directly: ``psd_plot`` /
  ``temp_plot`` / ``temp_plot_diff`` / ``pcs_plot``.
- ``utils`` --- the internal support layer the wrappers stand on, split by
  concern: ``util`` (config / dataset schema), ``io`` (pickle + raw loaders),
  ``signal`` (``correct_trigger`` / ``mne_epoch2raw`` / ``pearson_corr``),
  and re-exports of ``metric`` / ``log_or_print``. (File *lookup* is handled by
  the standalone ``osl_pathfinder.Pathfinder``.)

There is no ``batch`` re-export any more: every wrapper is registered as
``run_osl_<name>`` in ``osl_ephys.preprocessing.osl_wrappers``, so the plain
``osl_ephys.preprocessing.run_proc_batch`` resolves them by name -- import it
from there directly.

Group-level / source-space figure and statistics code (the old ``visualize``
subpackage: power maps, GLM cluster stats, parcel PSD, and the ``parcel_plot``
nibabel/nilearn surface helpers) no longer lives here --- it is project-level
analysis code, not part of the preprocessing package, and pulled heavy deps
(seaborn / glmtools / nilearn) into every ``import semp``.

The original ``semp`` package detected ``osl-ephys`` as an optional dep.
That guard is gone here --- this *is* osl-ephys --- so the import path is
unconditional and the legacy ``HAS_OSLE`` flag is no longer exported.
"""
from . import utils

# --- User-facing plotting ---
from .vis import psd_plot, temp_plot, temp_plot_diff, pcs_plot

# --- Support layer re-exports (live under utils; surfaced here for convenience) ---
from .utils import (
    pearson_corr,
    correct_trigger,
    mne_epoch2raw,
    SingletonEEG,
)
from .metric import EEGTracer, psd_band_ratio, psd_band_stat, mean_psd_in_band

# garbage.py is dead scratch code (never imported); not part of the package API.

# The project ``initialize`` extra_func is NOT provided by semp -- copy the
# template from the semp EEG-fMRI tutorial and adapt it to your acquisition.

# --- Preprocessing wrappers (resolved by name through osl-ephys find_func) ---
# (``apply_ica`` now lives in osl_ephys.preprocessing.manual_ica -- it belongs
# with the manual ICA review pipeline, not the automatic semp path.)
from .wrappers import (
    voltage_correction,
    cleanup,
    mid_crop,
    init_tracer,
    summary,
    ckpt_report,
    crop_TR,
    crop_by_epoch,
    create_epoch,
    create_TR_epoch,
    create_He_epoch,
    simulate_epoch,
    epoch_ssp,
    epoch_aas,
    epoch_obs,
    slice_reject,
    start_timer,
    end_timer,
)

# run_proc_batch / run_proc_chain are osl-ephys's own runners; re-exported here
# so existing `from ...semp import run_proc_batch` keeps working (see module
# docstring -- semp wrappers resolve by name, no shim needed).
from osl_ephys.preprocessing import run_proc_batch, run_proc_chain

__all__ = [
    "utils",
    "run_proc_batch",
    "run_proc_chain",
]
