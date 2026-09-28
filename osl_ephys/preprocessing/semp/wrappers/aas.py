import copy
import warnings

import numpy as np
import mne

from ..utils import (
    mne_epoch2raw,
    proc_userargs,
    require_keys,
    resolve_channel_names,
)


def epoch_aas(dataset, userargs):
    """Subtract a rolling TR template using one window or ``(short, long)``.

    With the default ``pre_pad=0.5``, the long template is used wherever its
    window can be centered on the target TR. The short template is used at the
    remaining recording edges; only its first/last ``(short - 1) // 2`` TRs
    then lack a centered window when both lengths are odd. A different
    ``pre_pad`` shifts both windows away from this default alignment.
    """
    userargs = proc_userargs(userargs, {
        'epoch_key': 'tr_ep',
        'window_length': 30,   # fastr defaults to 10 -- cleaner, but would notch the volume harmonics for volume trigger
        'picks': 'eeg',
        'overwrite': 'new',
        'fit': False,          # if False, standard AAS is used. if True, the avg template is fitted to the data first and then subtracted. (FASTR code style)
        'pre_pad': 0.5,        # in percentage, the padding before the first epoch. 1-pre_pad is the padding after the last epoch. (FASTR use pre_pad=0)
    })
    epoch_key = userargs['epoch_key']
    window_length = userargs['window_length']
    if isinstance(window_length, (tuple, list)):
        if len(window_length) != 2:
            raise ValueError("window_length must be an integer or (short, long).")
        short_length, long_length = window_length
        lengths = (short_length, long_length)
    else:
        lengths = (window_length,)

    if any(isinstance(length, (bool, np.bool_)) or
           not isinstance(length, (int, np.integer)) or length < 1
           for length in lengths):
        raise ValueError("window_length values must be positive integers.")
    lengths = tuple(int(length) for length in lengths)
    if len(lengths) == 2 and lengths[0] >= lengths[1]:
        raise ValueError("window_length must be ordered as (short, long).")
    if any(length % 2 == 0 for length in lengths):
        warnings.warn(
            "AAS is empirically found to perform sub-optimally when the target "
            "TR is not centered in the window. An even window_length makes "
            "exact centering impossible.",
            UserWarning,
            stacklevel=2,
        )
    require_keys(dataset, epoch_key, 'epoch_aas')
    picks = resolve_channel_names(
        dataset[epoch_key].info, userargs['picks']
    )
    overwrite = userargs['overwrite']
    fit = userargs['fit']
    pre_pad = userargs['pre_pad']
    if not 0 <= pre_pad <= 1:
        raise ValueError("pre_pad must be between 0 and 1.")

    orig_data = np.asarray(dataset[epoch_key].get_data(picks=picks))  # 29+#win, #ch, len(ep)
    n_epochs = len(orig_data)
    long_length = lengths[-1]
    if long_length > n_epochs:
        raise ValueError(
            "window_length {} exceeds the {} available epochs.".format(
                long_length, n_epochs
            )
        )
    # sliding window over epochs (np equivalent of torch's unfold(0, w, 1)):
    # appends the window axis as the last dim -> #win, #ch, len(ep), len(win)=#ep
    spurious_data = np.lib.stride_tricks.sliding_window_view(
        orig_data, long_length, axis=0)

    all_pcs = np.mean(spurious_data, axis=-1)[..., None]  # #win, #ch, len(ep), 1

    pre_padding = int(pre_pad * (long_length-1))
    post_padding = long_length - pre_padding - 1

    # pad the template ends by repeating the first/last window. np.repeat with
    # count 0 gives an empty array, so a zero-width side just drops out.
    pre = np.repeat(all_pcs[0:1], pre_padding, axis=0)
    post = np.repeat(all_pcs[-1:], post_padding, axis=0)
    all_pcs = np.concatenate([pre, all_pcs, post], axis=0)

    if len(lengths) == 2:
        short_length = lengths[0]
        short_pre = int(pre_pad * (short_length - 1))
        # Only the long-window edge epochs need the short template. Compute
        # those means directly so we do not allocate a second full-size
        # template array for a long, high-sampling-rate EEG recording.
        edge_indices = list(range(pre_padding)) + list(
            range(n_epochs - post_padding, n_epochs)
        )
        for index in edge_indices:
            start = min(max(index - short_pre, 0), n_epochs - short_length)
            all_pcs[index, ..., 0] = orig_data[start:start + short_length].mean(axis=0)

    if fit:
        # Least-squares fit of the single template's amplitude per epoch/channel.
        # For a one-column regressor this is exactly what lstsq computed:
        # alpha = <template, data> / <template, template>.
        tmpl = all_pcs[..., 0]                                     # 29+#win, #ch, len(ep)
        denom = np.sum(tmpl * tmpl, axis=-1, keepdims=True)
        alpha = np.sum(tmpl * orig_data, axis=-1, keepdims=True) / denom
        noise = alpha * tmpl
        cleaned = np.asarray(orig_data - noise)
    else:
        cleaned = np.asarray(orig_data - all_pcs[..., 0])  # squeeze last dim (safe for #ch==1)

    pc_name = f"pc_{epoch_key}"
    noise_name = f"noise_{epoch_key}"
    picks_name = f"picks_{epoch_key}"

    # To avoid overwriting existing keys, append underscores until a unique key is found.
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
    # Store the exact ordered names represented by the PC channel axis.  A
    # selector such as "all" is not stable enough for later alignment because
    # channel types and bad-channel state can change after this stage.
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)

    return dataset
