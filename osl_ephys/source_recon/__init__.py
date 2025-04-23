from .batch import *  # noqa: F401, F403
from .rhino.fsl_utils import setup_fsl, check_fsl  # noqa: F401, F403
from .freesurfer_utils import setup_freesurfer, check_freesurfer, recon_all, make_watershed_bem, make_fsaverage_src  # noqa: F401, F403
from .wrappers import find_template_subject  # noqa: F401, F403

with open(os.path.join(os.path.dirname(__file__), "README.md"), 'r') as f:
    __doc__ = f.read()