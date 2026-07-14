import copy

import numpy as np
import mne

from ..utils import proc_userargs, require_keys, mne_epoch2raw


def epoch_aas(dataset, userargs):
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
    picks = userargs['picks']
    overwrite = userargs['overwrite']
    fit = userargs['fit']
    pre_pad = userargs['pre_pad']

    require_keys(dataset, epoch_key, 'epoch_aas')
    orig_data = np.asarray(dataset[epoch_key].get_data(picks=picks))  # 29+#win, #ch, len(ep)
    # sliding window over epochs (np equivalent of torch's unfold(0, w, 1)):
    # appends the window axis as the last dim -> #win, #ch, len(ep), len(win)=#ep
    spurious_data = np.lib.stride_tricks.sliding_window_view(
        orig_data, window_length, axis=0)

    all_pcs = np.mean(spurious_data, axis=-1)[..., None]  # #win, #ch, len(ep), 1

    pre_padding = int(pre_pad * (window_length-1))
    post_padding = window_length - pre_padding - 1

    # pad the template ends by repeating the first/last window. np.repeat with
    # count 0 gives an empty array, so a zero-width side just drops out.
    pre = np.repeat(all_pcs[0:1], pre_padding, axis=0)
    post = np.repeat(all_pcs[-1:], post_padding, axis=0)
    all_pcs = np.concatenate([pre, all_pcs, post], axis=0)

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
    dataset[picks_name] = picks
    dataset['raw'] = mne_epoch2raw(dataset[epoch_key], dataset['raw'], cleaned, tmin=dataset[epoch_key].tmin, overwrite=overwrite, picks=picks)
    dataset[noise_name] = dataset[noise_name] - dataset['raw'].get_data()
    dataset[noise_name] = mne.io.RawArray(dataset[noise_name], dataset['raw'].info, first_samp=dataset['raw'].first_samp)

    return dataset
