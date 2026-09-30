# semp.utils --- the internal support layer the preprocessing wrappers stand on
# (not user-facing; for plotting you call directly, see semp.vis).
from .util import (  # config / dataset
    DATASET_SCHEMA,
    ensure_dir,
    proc_userargs,
    require_keys,
    resolve_channel_names,
)
from .io import load_pkl, save_pkl, SingletonEEG   # file io / loaders
from .signal import pearson_corr, correct_trigger, mne_epoch2raw            # EEG signal ops
