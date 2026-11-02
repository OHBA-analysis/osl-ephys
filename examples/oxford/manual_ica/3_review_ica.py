"""Label the ICA components of each session by hand.

The preprocessing fitted the ICA and marked the components MNE's EOG and
ECG detection flags without removing them (apply: false). This opens the
GUI on each session in turn: toggle the labels, close the window, and the
labelled components are removed from data/preproc/<id>_preproc-raw.fif.
The same from the terminal: osl-ephys ica review data --reject all
(one session: osl-ephys ica review data <id> --reject all).

The QC report's ICA tab shows the result; sessions can be reviewed again,
only the components labelled now being removed with reject="manual".
"""

from osl_ephys.preproc.ica import review_gui
from sessions import sessions

if __name__ == "__main__":
    for session in sessions:
        review_gui("data", session["id"], reject="all")
