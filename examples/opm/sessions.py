"""The sessions of the example: ten subjects with two runs each."""

sessions = [
    {
        "id": f"sub-{sub:03d}_run-{run:03d}",
        "subject": f"sub-{sub:03d}",
        "raw_file": f"data/raw/sub-{sub:03d}_run-{run:03d}_raw.fif",
        "smri_file": f"data/raw/sub-{sub:03d}_T1w.nii",
    }
    for sub in range(1, 11)
    for run in range(1, 3)
]
