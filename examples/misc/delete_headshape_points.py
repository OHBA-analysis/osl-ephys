"""Remove headshape points by hand and redo the coregistration.

After a batch run whose coregistration looks off for a session (the
coregistration tab of data/qc/report.html), open its headshape points,
delete the stray ones in the window (click a point, press d) and rerun the
coregistration from the saved points. Then rerun the session's source
reconstruction script, which recomputes the forward model.
"""

from osl_ephys.pipeline import run_batch
from osl_ephys.rhino.headshape import delete_headshape_points
from osl_ephys.session import Session

outdir = "data"
session = {
    "id": "sub-001_task-rest",
    "subject": "sub-001",
    "smri_file": "smri/sub-001_T1w.nii.gz",
}

files = Session(
    outdir=outdir,
    id=session["id"],
    surfaces_dir=f"{outdir}/surfaces/{session['subject']}",
    preproc_file=f"{outdir}/preproc/{session['id']}_preproc-raw.fif",
)
delete_headshape_points(files)

config = """
coreg:
  - coregister: {use_nose: false, use_headshape: true, n_init: 10}
"""
run_batch(config, [session], outdir, overwrite=True)
