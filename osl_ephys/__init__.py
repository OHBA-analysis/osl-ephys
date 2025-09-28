#!/usr/bin/python

# --------------------------------------------------------
# If user hasn't configured NumExpr environment the an irritating low-level log
# is generated. We supress it by setting a default value of 8 if not already
# set.
#
# https://numexpr.readthedocs.io/projects/NumExpr3/en/latest/user_guide.html#threadpool-configuration
# https://github.com/pydata/numexpr/blob/7c2ef387d81cd450e8220fe4174cf46ec559994c/numexpr/utils.py#L118

import os
if 'NUMEXPR_MAX_THREADS' not in os.environ:
    os.environ['NUMEXPR_MAX_THREADS'] = '8'

# Some modules are chatty by default when a logger is on - set log-levels to
# WARNING on setup
import logging
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

# Feels like there should be a better solution for this. How do we setup a
# logger which only produces OSL outputs?!

# --------------------------------------------------------
# Main importing - set module structure here

from . import utils  # noqa: F401, F403
from . import preprocessing  # noqa: F401, F403
from . import maxfilter  # noqa: F401, F403
from . import report  # noqa: F401, F403
from . import source_recon  # noqa: F401, F403
from . import glm  # noqa: F401, F403

# --------------------------------------------------------
osl_logger = logging.getLogger(__name__)
osl_logger.debug('osl-ephys main init complete')

# --------------------------------------------------------
__version__ = '2.4.0'
