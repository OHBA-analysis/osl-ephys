"""ICA wrappers for the semp preprocessing pipeline.

The interactive ``manual_ica`` review wrapper (and its HTML templates) lives in
``osl_ephys.preprocessing.manual_ica`` so it can be used by osl-ephys users who
don't depend on semp. We re-export ``manual_ica`` here so existing semp configs
(``{'manual_ica': {...}}``) keep working unchanged. Its companion ``apply_ica``
(applies a saved ICA at ``dataset['target_pth']/<subject>/<subject>_ica.fif``)
also lives there now -- it belongs with the manual review pipeline, not this
automatic path.

The one wrapper that remains inline is semp-specific:

* ``slice_reject`` --- flags the components of the **already-fitted** ICA
                       (``dataset['ica']`` from ``ica_raw``) that carry the
                       residual slice-timing artefact, and applies. Needs the
                       ``slice_interval`` / ``tr_interval`` keys that the
                       project ``initialize`` extra_func adds to ``dataset``.

Two ICA paths, end to end
-------------------------
There are two ways an ICA solution gets fitted, its bad components chosen, and
finally applied. They differ in *who fits* and *who chooses the bad ICs*:

    stage             | auto (sr_auto)                  | manual (sr_manual)
    ------------------|---------------------------------|---------------------------
    fit               | ica_raw -> dataset['ica']       | manual_ica (fits + saves
                      |                                 |   <subject>_ica.fif)
    choose bad ICs    | ica_autoreject(apply=False)     | human review in the
                      |   (EOG/ECG) + slice_reject      |   browser (label.txt)
                      |   (slice harmonics), unioned    |
                      |   into ica.exclude              |
    apply             | slice_reject (apply=True)       | apply_ica / osl-ica-apply
                      |   -- in-batch, this run's fit   |   (in manual_ica) --
                      |                                 |   reloads the saved ICA
    ICA lives in      | dataset['ica'] (memory)         | <subject>_ica.fif (disk)

The auto path reuses the single ``ica_raw`` fit (no second ICA); the manual path
round-trips through disk so a human can review between the fit and the apply.
"""
import numpy as np
import mne
from osl_ephys.utils.logger import log_or_print

from osl_ephys.preprocessing.manual_ica import manual_ica   # re-export; keeps semp configs working

from osl_ephys.preprocessing.semp.utils import proc_userargs, require_keys
from osl_ephys.preprocessing.semp.metric import mean_psd_in_band

__all__ = ['slice_reject', 'manual_ica']


def slice_reject(dataset, userargs):
    """Reject the ICA components carrying the residual slice-timing artefact.

    **Reuses the ICA already fitted by** ``ica_raw`` (in ``dataset['ica']``)
    rather than fitting a second one. For each component it scores the power at
    the slice-timing harmonics (``1/slice_interval`` and its multiples) against a
    local baseline band, and adds the components whose ratio exceeds
    ``noise2base_threshold`` to ``ica.exclude`` (a *union* with whatever
    ``ica_autoreject`` already marked as EOG/ECG). With ``apply=True`` (default)
    it then applies the ICA once, removing the EOG/ECG **and** slice components
    together.

    Run it after ``ica_raw`` + ``ica_autoreject``. Pair it with
    ``ica_autoreject(..., apply=False)`` so there is a single apply of the one
    fitted ICA::

        {'ica_raw': {...}},
        {'ica_autoreject': {..., 'apply': False}},   # mark EOG/ECG only
        {'slice_reject': {}},                         # add slice ICs, then apply

    Needs ``dataset['slice_interval']`` and ``dataset['tr_interval']`` (set by
    semp's ``initialize``). A per-recording override may be passed via
    ``dataset['slice_reject_n2b_threshold']`` (the old ``slice_ica_n2b_threshold``
    key is still honoured).
    """
    default_args = {
        'noise2base_threshold': 5.0,
        'noise_window': 0.5,
        'base_window': 5.0,
        'epoch_frange': [1, None],
        'apply': True,
    }
    userargs = proc_userargs(userargs, default_args)

    require_keys(dataset, ['ica', 'slice_interval', 'tr_interval'], 'slice_reject')

    noise2base_threshold = userargs['noise2base_threshold']
    noise_window = userargs['noise_window']
    base_window = userargs['base_window']
    epoch_frange = userargs['epoch_frange']

    # per-recording override from initialize() (new key preferred, old honoured)
    for key in ('slice_reject_n2b_threshold', 'slice_ica_n2b_threshold'):
        if key in dataset:
            noise2base_threshold = dataset[key]
            log_or_print(f"slice_reject: noise2base_threshold={noise2base_threshold} "
                         f"from dataset[{key!r}]")
            break

    assert base_window > noise_window, 'base_window should be greater than noise_window.'

    slice_freq = 1 / dataset['slice_interval']
    # tr_freq sets the *width* of the noise/base bands around each slice
    # harmonic (noise_window * tr_freq / 2 Hz): the gradient artefact's
    # sidebands sit at multiples of the volume (TR) frequency, so the band that
    # should capture a harmonic's peak scales with 1/TR. This is why slice_reject
    # needs tr_interval. (To decouple, express noise_window/base_window in Hz.)
    tr_freq    = 1 / dataset['tr_interval']

    ica = dataset['ica']
    data = ica.get_sources(dataset['raw'])._data
    psds, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=dataset['raw'].info['sfreq'],
        fmin=epoch_frange[0],
        fmax=epoch_frange[1] if epoch_frange[1] is not None else dataset['raw'].info['sfreq']/ 2,
        n_fft=int(round(dataset['raw'].info['sfreq'] * 20)),
    )

    eps = 1e-10
    harmonics = np.arange(slice_freq, freqs.max(), slice_freq)
    slice_ics = []
    for ic in range(data.shape[0]):
        psd_row = psds[ic]
        for harmonic in harmonics:
            noise = mean_psd_in_band(psd_row, freqs, harmonic, noise_window * tr_freq / 2)
            base  = mean_psd_in_band(psd_row, freqs, harmonic, base_window  * tr_freq / 2)
            base = (base * base_window - noise * noise_window) / (base_window - noise_window)
            if (noise / (base + eps)) > noise2base_threshold:
                slice_ics.append(ic)
                break

    # union with whatever ica_autoreject already marked (EOG/ECG)
    ica.exclude = sorted(set(ica.exclude) | set(slice_ics))
    log_or_print(f"slice_reject: {len(slice_ics)} slice-harmonic IC(s) {slice_ics}; "
                 f"ica.exclude now {ica.exclude}")

    if userargs['apply']:
        dataset['raw'] = ica.apply(dataset['raw'].copy())
    return dataset
