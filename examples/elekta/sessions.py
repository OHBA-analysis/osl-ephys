"""The sessions of the example: their ids, subjects and raw files.

A session is a dict with an ``id`` and the files its steps need (``raw_file``
for preprocessing, ``smri_file`` for the surfaces); a CSV file with the same
columns can be passed to ``run_batch`` instead.
"""

subjects = ["sub-001", "sub-002"]

sessions = [
    {
        "id": f"{subject}_task-rest",
        "subject": subject,
        "raw_file": f"raw/{subject}_task-rest.fif",
        "smri_file": f"smri/{subject}_T1w.nii.gz",
    }
    for subject in subjects
]
