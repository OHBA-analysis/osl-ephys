"""Align the sign of each parcel time course across sessions.

The group step picks the template session (the one whose time-embedded
covariance is most like the others') and flips every session to match it,
writing data/<id>/<id>_sflip_parc-raw.fif. Only needed before fitting a
group model on time-delay embedded data.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
group:
  - sign_flip: {n_embeddings: 15, standardize: true}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data")
