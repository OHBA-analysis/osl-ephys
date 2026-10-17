"""The sessions of the example."""

subjects = ["sub-001", "sub-002"]

sessions = [
    {
        "id": f"{subject}_task-rest",
        "subject": subject,
        "raw_file": f"raw/{subject}/{subject}_task-rest_meg.ds",
        "pos_file": f"raw/{subject}/{subject}_headshape.pos",
        "smri_file": f"smri/{subject}_T1w.nii",
    }
    for subject in subjects
]
