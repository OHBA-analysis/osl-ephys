# Pipelines for data collected at OHBA, Oxford

Continuous recordings from the OHBA TRIUX (Neo), at rest or during a task,
from the raw `.fif` files to sign-flipped parcel time courses. The
pipelines differ in how the sensor-level data is cleaned after
preprocessing, before source reconstruction:

- `automated_ica/`: ICA with the components labelled automatically. Use
  this for a first pass; it needs the least intervention.
- `manual_ica/`: ICA with the components labelled by hand in the GUI.
- `ssp/`: signal-space projection instead of ICA.
- `maxfilter/`: MaxFilter alone, and which scanner option to use for
  recordings from the older OHBA scanners.

Each pipeline is a directory of numbered scripts with its sessions in
`sessions.py`; see `../README.md` for how the scripts are laid out.
