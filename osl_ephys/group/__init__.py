"""Group-level steps: sign flipping of the parcel time courses across
sessions."""

from osl_ephys.group.sign_flipping import (
    DEFAULT_METHOD,
    METHODS,
    find_flips,
    get_method_hyperparameters,
    sign_flip,
    sign_flip_mne_raw,
)
