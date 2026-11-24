"""Sign flipping of parcel time courses held in MATLAB files.

The pipeline's sign_flip step does this for parcellated fif files; here the
same functions run on (time, parcels) arrays loaded from elsewhere.
"""

import numpy as np
from scipy import io

from osl_ephys.group import sign_flipping

files = [f"src/subject{i}.mat" for i in range(1, 11)]
n_embeddings, standardize = 15, True

data = [io.loadmat(f)["X"] for f in files]
covs = np.stack(
    [sign_flipping.compute_covariance(x, n_embeddings, standardize) for x in data]
)

# The template is a typical subject: the one with the median total similarity
# of its (sign-invariant) covariance to the others'
template = sign_flipping.find_template(covs, diag_offset=n_embeddings)
print("Template:", files[template])

for i, (x, cov) in enumerate(zip(data, covs)):
    if i == template:
        flipped = x
    else:
        # method can be "sdp" (the default), "spectral", "belief_propagation" or
        # "random", followed by that method's options
        flips, corr = sign_flipping.find_flips(
            cov, covs[template], n_embeddings, method="sdp"
        )
        print(f"{files[i]}: correlation with the template {corr:.3f}")
        flipped = x * flips[np.newaxis, :]
    io.savemat(f"src/sflip{i + 1}.mat", {"X": flipped, "T": flipped.shape[0]})
