# Example scripts

Each directory is a pipeline from the raw recordings to sign-flipped
parcel time courses, as numbered scripts. Every script has the same shape:
the sessions (a list of dicts in `sessions.py`, or a CSV file with the
same columns) and a config at the top, then `run_batch`, which runs the
config on every session in parallel, writes a log per session and the QC
report (`data/qc/report.html`). The stages are separate scripts so the
report can be checked between them: MaxFilter (on the computer with the
licence; the preprocessing script carries the same section, which is
skipped once the output exists), preprocessing, coregistration (surfaces
and coregistration, the stage to rerun when it looks off in the report),
source reconstruction (forward model, beamforming and parcellation), and
the sign flipping across sessions. Their sections can also be put in one
config and run in one go.

- `elekta/`, `ctf/`, `eeg/`, `opm/`: one pipeline per scanner type.
- `oxford/`: the pipelines for data from the OHBA scanner, differing in
  how the sensor data is cleaned (automatic ICA, manual ICA, SSP), plus
  MaxFilter alone.
- `wakeman_henson/`: the analysis of the osl-ephys toolbox paper on the
  face recognition dataset ds000117, including the statistics, and the
  FreeSurfer-based route as an alternative.
- `misc/`: smaller scripts for particular situations.
