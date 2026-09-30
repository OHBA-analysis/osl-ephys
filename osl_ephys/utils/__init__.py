#!/usr/bin/python

from . import simulate  ## noqa: F401, F403
from . import logger  ## noqa: F401, F403
from .study import Study  # noqa: F401, F403
from .file_handling import *  # noqa: F401, F403
from .spmio import SPMMEEG  # noqa: F401, F403
from .parallel import dask_parallel_bag  # noqa: F401, F403
from .simulate import *  # noqa: F401, F403
from .opm import *  # noqa: F401, F403
from .package import soft_import, run_package_tests  # noqa: F401, F403
from .misc import ensure_dir, proc_userargs, mean_psd_in_band  # noqa: F401, F403
from .version_utils import check_version # noqa: F401, F403
from . import run_func  # noqa: F401, F403

with open(os.path.join(os.path.dirname(__file__), "README.md"), 'r') as f:
    __doc__ = f.read()
