"""LCMV beamforming on the MNI grid of the forward model."""

from osl_ephys.source_recon.beamforming.lcmv import (
    lcmv_beamformer,
    apply_lcmv_beamformer,
    virtual_electrodes,
    lcmv_weights,
    leadfields,
)
from osl_ephys.source_recon.beamforming.bilateral import plot_bilateral_pairs
