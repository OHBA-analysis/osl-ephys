"""Align the sign of each parcel time course across sessions.

The group step picks the template session and flips every other
session to match it (with osl-signflip's sdp method by default),
writing data/<id>/<id>_sflip-lcmv-parc-raw.fif. Only needed before
fitting a group model on time-delay embedded data.

The flips are found with osl-signflip; method can be sdp (the default),
spectral, belief_propagation or random, followed by that method's options.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
group:
  - sign_flip: {n_embeddings: 15, standardize: true, method: sdp}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data")
