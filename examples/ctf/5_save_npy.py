"""Save the sign-flipped parcel time courses as numpy arrays (time, parcels)
with osl-dynamics, ready for training a model."""

from glob import glob

from osl_dynamics.data import Data

files = sorted(glob("data/*/*_sflip_parc-raw.fif"))
data = Data(files, picks="misc", reject_by_annotation="omit", n_jobs=4)
data.save("data/npy")
data.delete_dir()
