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
covs = [sign_flipping.calc_cov(x, n_embeddings, standardize) for x in data]

# The template is the subject whose covariance is most like the others'
scores = [
    np.mean([sign_flipping.calc_corr(a, b, n_embeddings) for b in covs if b is not a])
    for a in covs
]
template = int(np.argmax(scores))
print("Template:", files[template])

for i, (x, cov) in enumerate(zip(data, covs)):
    if i == template:
        flipped = x
    else:
        flips, corr = sign_flipping.find_flips(
            cov, covs[template], x.shape[1], n_embeddings
        )
        print(f"{files[i]}: correlation with the template {corr:.3f}")
        flipped = x * flips[np.newaxis, :]
    io.savemat(f"src/sflip{i + 1}.mat", {"X": flipped, "T": flipped.shape[0]})
