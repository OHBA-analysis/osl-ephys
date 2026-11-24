"""Align the sign of each parcel time course across sessions.

Writes data/<id>/<id>_sflip-lcmv-parc-raw.fif, or <id>_sflip-eloreta-parc-raw.fif
after the FreeSurfer route.
"""

from osl_ephys.pipeline import run_batch
from sessions import sessions

config = """
group:
  - sign_flip: {n_embeddings: 15, standardize: true}
"""

if __name__ == "__main__":
    run_batch(config, sessions, "data")
