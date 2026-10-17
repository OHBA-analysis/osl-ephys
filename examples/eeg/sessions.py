"""The sessions of the example."""

subjects = ["sub-001", "sub-002"]

sessions = [
    {
        "id": f"{subject}_task-rest",
        "subject": subject,
        "raw_file": f"raw/{subject}_task-rest_eeg.vhdr",
        "smri_file": f"smri/{subject}_T1w.nii.gz",
    }
    for subject in subjects
]
