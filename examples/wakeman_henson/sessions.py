"""The 19 subjects and 6 runs of ds000117.

The ids are ``sub001_run-01`` to ``sub019_run-06``; the surfaces are
extracted once per subject from its structural MRI.
"""

subjects = [f"sub{i:03d}" for i in range(1, 20)]

sessions = [
    {
        "id": f"{subject}_run-{run:02d}",
        "subject": subject,
        "raw_file": f"ds117/{subject}/MEG/run_{run:02d}_raw.fif",
        "smri_file": f"ds117/{subject}/anatomy/highres001.nii.gz",
    }
    for subject in subjects
    for run in range(1, 7)
]
