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
from osl_ephys.utils.logger import log_or_print

from osl_ephys.preprocessing.manual_ica import manual_ica   # re-export; keeps semp configs working
from osl_ephys.preprocessing.manual_ica.helpers import (
    _compute_slice_contribution_scores,
    _compute_slice_ga_scores,
    _good_mask,
)

from osl_ephys.preprocessing.semp.utils import proc_userargs, require_keys

__all__ = ['slice_reject', 'manual_ica']


def slice_reject(dataset, userargs):
    """Reject the ICA components carrying the residual slice-timing artefact.

    **Reuses the ICA already fitted by** ``ica_raw`` (in ``dataset['ica']``)
    rather than fitting a second one. For each component and each slice-timing
    harmonic (``1/slice_interval`` and multiples), it scores the largest PSD
    bin against both its local spectral shoulders and the strongest
    non-harmonic 5-45 Hz peak. The same harmonic must exceed both thresholds.
    An optional contribution gate can additionally rescue a component which
    accounts for a large fraction of the still-unhandled sensor-level
    harmonic excess despite being spectrally mixed with other activity.
    It adds those components to
    ``ica.exclude`` (a *union*
    with whatever ``ica_autoreject`` already marked as EOG/ECG). With
    ``apply=True`` (default)
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
    ``dataset['slice_reject_local_threshold']`` and
    ``dataset['slice_reject_dominance_threshold']``.
    """
    default_args = {
        'local_threshold': 8.0,
        'dominance_threshold': 1.0,
        'fmin': 1.0,
        'fmax': 45.0,
        'dominance_fmin': 5.0,
        'peak_window': 1.0,
        'base_window': 5.0,
        'contribution_gate': False,
        'contribution_local_threshold': 4.0,
        'contribution_fraction_threshold': 0.30,
        'apply': True,
    }
    userargs = proc_userargs(userargs, default_args)

    require_keys(dataset, ['ica', 'slice_interval', 'tr_interval'], 'slice_reject')

    local_threshold = userargs['local_threshold']
    dominance_threshold = userargs['dominance_threshold']
    peak_window = userargs['peak_window']
    base_window = userargs['base_window']

    if 'slice_reject_local_threshold' in dataset:
        local_threshold = dataset['slice_reject_local_threshold']
        log_or_print(
            'slice_reject: local_threshold='
            f'{local_threshold} from '
            "dataset['slice_reject_local_threshold']"
        )
    if 'slice_reject_dominance_threshold' in dataset:
        dominance_threshold = dataset['slice_reject_dominance_threshold']
        log_or_print(
            'slice_reject: dominance_threshold='
            f'{dominance_threshold} from '
            "dataset['slice_reject_dominance_threshold']"
        )

    if base_window <= peak_window or peak_window <= 0:
        raise ValueError('base_window must be greater than peak_window > 0.')

    ica = dataset['ica']
    data = ica.get_sources(dataset['raw']).get_data()
    scores = _compute_slice_ga_scores(
        data,
        sfreq=dataset['raw'].info['sfreq'],
        slice_interval=dataset['slice_interval'],
        tr_interval=dataset['tr_interval'],
        good_mask=_good_mask(dataset['raw']),
        fmin=userargs['fmin'],
        fmax=userargs['fmax'],
        dominance_fmin=userargs['dominance_fmin'],
        peak_window=peak_window,
        base_window=base_window,
        local_threshold=local_threshold,
        dominance_threshold=dominance_threshold,
    )
    dataset['slice_reject_scores'] = scores

    local_scores = scores['peak_to_local']
    dominance_scores = scores['peak_to_elsewhere']
    slice_ics = [
        int(ic) for ic in np.flatnonzero(
            (local_scores > local_threshold)
            & (dominance_scores > dominance_threshold)
        )
    ]

    contribution_ics = []
    if userargs['contribution_gate']:
        already_excluded = sorted(set(ica.exclude) | set(slice_ics))
        contribution = _compute_slice_contribution_scores(
            scores,
            ica.get_components(),
            excluded=already_excluded,
            local_threshold=userargs['contribution_local_threshold'],
            fraction_threshold=userargs['contribution_fraction_threshold'],
        )
        contribution_ics = [
            int(ic) for ic in np.flatnonzero(contribution['passes'])
        ]
        scores['contribution_fraction'] = contribution['fraction']
        scores['contribution_local'] = contribution['local']
        scores['contribution_harmonic_hz'] = contribution['harmonic_hz']
        scores['contribution_passes'] = contribution['passes']
        contribution_details = [
            f'{ic} (harmonic={contribution["harmonic_hz"][ic]:.3f} Hz, '
            f'local={contribution["local"][ic]:.2f}, '
            f'fraction={contribution["fraction"][ic]:.3f})'
            for ic in contribution_ics
        ]
        log_or_print(
            'slice_reject: contribution-gate rescue '
            f'{contribution_details}'
        )

    # union with whatever ica_autoreject already marked (EOG/ECG)
    ica.exclude = sorted(
        set(ica.exclude) | set(slice_ics) | set(contribution_ics)
    )
    scored_ics = [
        f'{ic} (harmonic={scores["harmonic_hz"][ic]:.3f} Hz, '
        f'local={local_scores[ic]:.2f}, '
        f'dominance={dominance_scores[ic]:.2f})'
        for ic in slice_ics
    ]
    log_or_print(
        'slice_reject: '
        f'{len(slice_ics)} slice-harmonic IC(s), two-gate scores '
        f'{scored_ics}; ica.exclude now {ica.exclude}'
    )

    if userargs['apply']:
        dataset['raw'] = ica.apply(dataset['raw'].copy())
    return dataset
