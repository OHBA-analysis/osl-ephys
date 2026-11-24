"""The ``maxfilter`` section of a config: one step, run on the session's raw
file before preprocessing.

The step writes the maxfiltered file under ``outdir/maxfilter/`` and the
``preproc`` section reads it in place of the raw file. A config with only a
``maxfilter`` section is the usual way to run it on the machine that has the
MaxFilter licence.
"""

from __future__ import annotations

import logging
from pathlib import Path

from osl_ephys.pipeline.registry import step

logger = logging.getLogger("osl_ephys")


@step("maxfilter")
def maxfilter(dataset, userargs):
    """Run MaxFilter on the session's ``raw_file``
    (:func:`~osl_ephys.maxfilter.run_maxfilter`): the options are its
    keyword arguments (``mode``, ``scanner``, ``tsss``, ``headpos``,
    ``movecomp``, ``trans``, ...). The output goes to ``outdir/maxfilter``
    and becomes the session's data for the sections that follow.
    """
    from osl_ephys.maxfilter.maxfilter import output_file, run_maxfilter

    session = dataset["session"]
    if "raw_file" not in session:
        raise ValueError(f"Session '{session['id']}' has no raw_file to maxfilter.")
    outdir = Path(dataset["paths"]["session_dir"]).parent / "maxfilter"
    outdir.mkdir(parents=True, exist_ok=True)
    userargs.pop("target", None)
    run_maxfilter([str(session["raw_file"])], str(outdir), **userargs)
    dataset["maxfilter_file"] = output_file(
        str(session["raw_file"]),
        str(outdir),
        mode=userargs.get("mode", "standard"),
        tsss=userargs.get("tsss", False),
        trans=userargs.get("trans"),
    )
    logger.info(f"Maxfiltered file: {dataset['maxfilter_file']}")
    return dataset
