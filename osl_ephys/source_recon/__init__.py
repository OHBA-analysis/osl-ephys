from .batch import *  # noqa: F401, F403
from .rhino.fsl_utils import setup_fsl  # noqa: F401, F403
from .freesurfer_utils import setup_freesurfer  # noqa: F401, F403
from .wrappers import find_template_subject  # noqa: F401, F403

with open(os.path.join(os.path.dirname(__file__), "README.md"), 'r') as f:
    __doc__ = f.read()