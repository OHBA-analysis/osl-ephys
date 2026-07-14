import copy

import numpy as np
import mne

from ..utils import proc_userargs, require_keys, mne_epoch2raw


def _lower_median(x, axis=None):
    """Match torch.median: the lower of the two middle values for even counts.

    numpy's ``np.median`` averages the two central values; torch returns the
    lower one. We replicate torch here so the high-power screening threshold is
    unchanged from the original implementation.
    """
    x = np.asarray(x)
    n = x.shape[axis] if axis is not None else x.size
    k = (n - 1) // 2
    part = np.sort(x, axis=axis)
    if axis is None:
        return part[k]
    return np.take(part, k, axis=axis)


def epoch_obs(dataset, userargs):
    userargs = proc_userargs(userargs, {
        'epoch_key': 'tr_ep',
        'npc': 3,
        'picks': 'eeg',
        'overwrite': 'even',
        # remove_mean: subtract each epoch's per-channel mean before the SVD
        # (standard PCA centering). Set it by the EPOCH LENGTH:
        #   * True  for TR- or heartbeat-length epochs (tr_ep / he_ep): centering
        #           is safe, and without it we have seen step noise appear at the
        #           epoch borders.
        #   * False for very short slice-length epochs (slice_ep, <~0.1 s): the
        #           mean of so few samples IS real low-frequency signal, so
        #           removing it eats brain data. This is why Niazy's slice-wise
        #           OBS does not center (unlike standard PCA).
        'remove_mean': True,
        'pc_from_spurious': True,   # if True, the PC is calculated from all events, else it is calculated from the safe epochs. This parameter is only used for BCG correction, where the heartbeat detection could mistake residual GA / motion as heartbeats.
        'apply_to_spurious': True,  # if True, the PC is applied to the spurious events, else it is not.
        'screen_high_power': None,  # if True, the epochs with high power would not be used for PC calculation. If None, no screening is performed. If false, only the epochs with high power would be used for PC calculation.
    })
    epoch_key = userargs['epoch_key']
    npc = userargs['npc']
    picks = userargs['picks']
    overwrite = userargs['overwrite']
    remove_mean = userargs['remove_mean']
    pc_from_spurious = userargs['pc_from_spurious']
    apply_to_spurious = userargs['apply_to_spurious']
    screen_high_power = userargs['screen_high_power']

    require_keys(dataset, epoch_key, 'epoch_obs')
    if pc_from_spurious:
        orig_data = np.asarray(dataset[epoch_key].get_data(picks=picks))  # #ep, #ch, len(ep)
    else:
        orig_data = np.asarray(dataset[f"{epoch_key}_safe"].get_data(picks=picks))

    if screen_high_power is not None:
        epoch_power = np.sum(orig_data**2, axis=(1, 2))  # #ep
        power_med = _lower_median(epoch_power)
        power_mad = _lower_median(np.abs(epoch_power - power_med))
        threshold = power_med + 3*power_mad
        orig_data = orig_data[epoch_power < threshold] if screen_high_power else orig_data[epoch_power >= threshold]

    orig_data = np.transpose(orig_data, (1, 2, 0))  # #ch, len(ep), #ep

    pca_mean = np.mean(orig_data, axis=1) * int(remove_mean)    # #ch, #ep
    dirty_data = orig_data - pca_mean[:, None, :]
    # batched SVD over the #ch axis (np equivalent of torch.linalg.svd):
    U, S, _ = np.linalg.svd(dirty_data, full_matrices=False)  # #ch, len(ep), K;  #ch, K;  #ch, K, #ep
    all_pcs = U[..., :npc] * S[..., None, :npc]

    del orig_data, dirty_data, U, S  # free memory
    if apply_to_spurious:
        orig_data = np.asarray(dataset[epoch_key].get_data(picks=picks))  # #ep, #ch, len(ep)
    else:
        orig_data = np.asarray(dataset[f"{epoch_key}_safe"].get_data(picks=picks))
    orig_data = np.transpose(orig_data, (1, 2, 0))  # #ch, len(ep), #ep
    pca_mean = np.mean(orig_data, axis=1) * int(remove_mean)    # #ch, #ep
    dirty_data = orig_data - pca_mean[:, None, :]  # #ch, len(ep), #ep
    # least-squares projection of dirty_data onto the OBS basis (all_pcs), via
    # the normal equations -- the batched equivalent of torch.linalg.lstsq.
    # gram is (#ch, npc, npc) (tiny, well-conditioned: ~diag(S**2)).
    at = np.swapaxes(all_pcs, -1, -2)              # #ch, #pc, len(ep)
    gram = at @ all_pcs                            # #ch, #pc, #pc
    rhs = at @ dirty_data                          # #ch, #pc, #ep
    coef = np.linalg.solve(gram, rhs)              # #ch, #pc, #ep
    noise = all_pcs @ coef + pca_mean[:, None, :]  # #ch, len(ep), #ep

    cleaned = np.ascontiguousarray(np.transpose(orig_data - noise, (2, 0, 1)))

    pc_name = f"pc_{epoch_key}"
    noise_name = f"noise_{epoch_key}"
    picks_name = f"picks_{epoch_key}"

    # To avoid overwriting existing keys (e.g. if AAS already ran), append underscores.
    while True:
        if pc_name in dataset:
            pc_name = pc_name + "_"
            continue
        if noise_name in dataset:
            noise_name = noise_name + "_"
            continue
        if picks_name in dataset:
            picks_name = picks_name + "_"
            continue
        break

    dataset[noise_name] = copy.deepcopy(dataset['raw'].get_data())
    dataset[pc_name] = all_pcs
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)
    return dataset
