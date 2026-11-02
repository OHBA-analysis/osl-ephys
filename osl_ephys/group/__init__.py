"""Group-level steps: sign flipping of the parcel time courses across
sessions."""

from osl_ephys.group.sign_flipping import (
    find_flips,
    sign_flip,
    sign_flip_mne_raw,
    find_flips_spectral,
    find_flips_belief_propagation,
    find_flips_sdp,
)
