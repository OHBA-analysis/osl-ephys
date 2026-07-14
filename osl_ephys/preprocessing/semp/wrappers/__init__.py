# Wrapper naming convention (follow it when adding the next wrapper):
#   * <verb>_<noun>  for a stage that transforms the dataset/raw --
#       create_epoch, create_TR_epoch, simulate_epoch, crop_TR, crop_by_epoch,
#       mid_crop, slice_reject, voltage_correction, init_tracer.
#   * <noun>_<method> for the artefact-template family, read as "<epoch>-wise
#       <method>" -- epoch_aas, epoch_obs, epoch_ssp.
# (start_timer/end_timer/cleanup/summary/ckpt_report are standalone verbs.)
# Renaming existing wrappers is a breaking change -- every config references
# them by name and osl_wrappers.py builds a run_osl_<name> stub per name -- so
# this governs *new* wrappers, not a churn of the current ones.

from .misc import voltage_correction, cleanup, mid_crop
from .report import init_tracer, summary, ckpt_report
from .epoching import crop_TR, crop_by_epoch, create_epoch, create_TR_epoch, create_He_epoch, simulate_epoch
from .ssp import epoch_ssp
from .aas import epoch_aas
from .obs import epoch_obs
from .ica import slice_reject, manual_ica
from .timer import start_timer, end_timer
