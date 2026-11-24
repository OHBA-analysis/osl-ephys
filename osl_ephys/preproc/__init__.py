"""Sensor-level preprocessing.

The steps of the ``preproc`` section of a config (:mod:`~osl_ephys.preproc.steps`):
MNE methods on Raw and Epochs (``filter``, ``resample``, ``epochs``, ...),
the detectors for bad segments, channels and epochs, ICA with automatic,
correlation-based or manual labelling, headshape decimation and the GLM
steps. The functions behind them are in :mod:`~osl_ephys.preproc.functions`,
the reading and writing of a session's files in
:mod:`~osl_ephys.preproc.dataset` and the QC plots in
:mod:`~osl_ephys.preproc.qc`.
"""

from osl_ephys.preproc.functions import (
    gesd,
    detect_artefacts,
    detect_bad_segments,
    detect_bad_channels,
    detect_bad_channels_psd,
    detect_maxfilt_zeros,
    drop_bad_epochs,
    ica_label,
    ica_ecg_eog_correlation,
    decimate_headshape_points,
)
from osl_ephys.preproc.dataset import (
    import_data,
    read_dataset,
    write_dataset,
    plot_preproc_flowchart,
)
