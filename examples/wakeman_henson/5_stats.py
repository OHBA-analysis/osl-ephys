"""Statistics on the parcel time courses.

The sign-flipped parcel files are the input: each session is epoched
around the face presentations and a GLM fitted with one regressor per
face type. The group section fits a GLM over the sessions with one
regressor per subject, and tests the mean response between 50 and 300 ms
with a max-statistic permutation test. The group results are pickled in
data/stats.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

stats_sessions = [
    {
        "id": session["id"],
        "subject": session["subject"],
        "raw_file": f"data/{session['id']}/{session['id']}_sflip_parc-raw.fif",
    }
    for session in sessions
]
covs = {"Subject": [session["subject"] for session in sessions]}

config = """
meta:
  event_codes:
    famous/first: 5
    famous/immediate: 6
    famous/last: 7
    unfamiliar/first: 13
    unfamiliar/immediate: 14
    unfamiliar/last: 15
    scrambled/first: 17
    scrambled/immediate: 18
    scrambled/last: 19
preproc:
  - find_events: {min_duration: 0.005}
  - epochs: {picks: misc, tmin: -0.2, tmax: 0.3}
  - glm_add_regressor: {name: famous, rtype: Categorical, codes: [5 6 7]}
  - glm_add_regressor: {name: unfamiliar, rtype: Categorical, codes: [13 14 15]}
  - glm_add_regressor: {name: scrambled, rtype: Categorical, codes: [17 18 19]}
  - glm_add_contrast: {name: Mean, values: {famous: 1/3, unfamiliar: 1/3, scrambled: 1/3}}
  - glm_add_contrast: {name: Faces-Scrambled, values: {famous: 1, unfamiliar: 1, scrambled: -2}}
  - glm_fit: {method: epochs}
group:
  - glm_add_regressor: {name: Subject, rtype: Categorical, key: Subject, codes: unique}
  - glm_add_contrast: {name: Mean, values: unique, key: Subject}
  - glm_fit: {method: epochs}
  - glm_permutations: {method: epochs, type: max, contrast: Mean, nperms: 1000, threshold: 0.99, tmin: 0.05, tmax: 0.3}
"""

if __name__ == "__main__":
    run_batch(
        config, stats_sessions, "data/stats", n_workers=16, covs=covs, qc_dir=False
    )
