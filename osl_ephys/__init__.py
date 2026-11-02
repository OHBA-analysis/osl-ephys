from importlib.metadata import PackageNotFoundError, version

from . import utils
from . import io
from . import preproc
from . import maxfilter
from . import report
from . import rhino
from . import source_recon
from . import parcellation
from . import group
from . import glm

# The version number comes from the installed package metadata, which
# setuptools_scm derives from the git tag
try:
    __version__ = version("osl-ephys")
except PackageNotFoundError:
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError
