import mne

from ..utils import proc_userargs, require_keys


def epoch_ssp(dataset, userargs):
    userargs = proc_userargs(userargs, {
        'ssp': 0,
        'epoch_key': 'tr_ep',
        'apply': False,   # whether to apply all projections including the SSP.
    })
    ssp = userargs['ssp']
    epoch_key = userargs['epoch_key']
    apply = userargs['apply']

    require_keys(dataset, epoch_key, 'epoch_ssp')
    proj = mne.compute_proj_epochs(dataset[epoch_key], n_grad=0, n_mag=0, n_eeg=ssp, verbose=True)
    dataset['raw'].add_proj(proj)

    if apply:
        dataset['raw'].apply_proj()

    # TODO: add option to save the SSP components & noise in the dataset for visualization in the ckpt_report step.
    return dataset
