"""Parcellation: parcel time courses from source-space data, and the
parcellation files."""

from osl_ephys.parcellation.parcellation import (
    Parcellation,
    parcellate,
    parcellate_lcmv,
    plot_parcellation,
    load_parcellation,
    load_parcellation_description,
    guess_parcellation,
    parcel_centers,
    convert_to_mne_raw,
    save_as_fif,
)
