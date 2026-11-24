"""The config API: one YAML config run over a set of sessions.

Modules
-------
- :py:mod:`~osl_ephys.pipeline.config` — Loading and checking a config.
- :py:mod:`~osl_ephys.pipeline.registry` — The steps a config can name.
- :py:mod:`~osl_ephys.pipeline.manifest` — The sessions and their files.
- :py:mod:`~osl_ephys.pipeline.chain` — Run a config on one session.
- :py:mod:`~osl_ephys.pipeline.batch` — Run a config on every session, in
  parallel.
- :py:mod:`~osl_ephys.pipeline.runner` — The parallel runner, with a log
  file per session and a summary of the run.
"""

from osl_ephys.pipeline.batch import run_batch
from osl_ephys.pipeline.chain import run_chain
from osl_ephys.pipeline.config import load_config
