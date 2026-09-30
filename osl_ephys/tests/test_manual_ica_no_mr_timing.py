"""Manual ICA review also works for ordinary EEG without MR timing."""

import importlib
import json
import re

import mne
import numpy as np
import pytest


@pytest.mark.parametrize('with_aux, ecg_threshold', [
    (False, 'auto'), (True, 'auto'), (True, 0.1),
])
def test_manual_ica_without_mr_timing_omits_ga_scores(
    tmp_path, monkeypatch, with_aux, ecg_threshold,
):
    ica_module = importlib.import_module('osl_ephys.preprocessing.manual_ica.ica')
    rng = np.random.RandomState(7)
    names = ['Fz', 'Cz', 'Pz', 'Oz'] + (['EOGU', 'EOGL', 'ECG'] if with_aux else [])
    types = ['eeg'] * 4 + (['eog', 'eog', 'ecg'] if with_aux else [])
    info = mne.create_info(names, 250, types)
    raw = mne.io.RawArray(rng.standard_normal((len(names), 5000)) * 1e-6, info,
                          verbose=False)
    raw.set_montage('standard_1020')

    # Leave scoring and HTML generation real; skip only expensive SVG renders.
    for name in ('_plot_ic_summary', '_render_ic_topo', '_render_ic_zoom',
                 '_render_zoom_clean'):
        monkeypatch.setattr(ica_module, name, lambda *args, **kwargs: None)

    expected_ecg_threshold = 0.32 if ecg_threshold == 'auto' else ecg_threshold
    if with_aux:
        def find_bads_eog(self, raw, ch_name, measure, threshold):
            assert ch_name == ['EOGU', 'EOGL']
            assert measure == 'correlation'
            assert threshold == 0.35
            return [0], [np.array([-0.915, 0.2]), np.array([0.5585, 0.1])]
        monkeypatch.setattr(mne.preprocessing.ICA, 'find_bads_eog', find_bads_eog)

        def find_bads_ecg(self, raw, ch_name, method, threshold):
            assert ch_name == 'ECG'
            assert method == 'ctps'
            assert threshold == expected_ecg_threshold
            return [0], np.array([threshold, threshold * 0.8])
        monkeypatch.setattr(mne.preprocessing.ICA, 'find_bads_ecg', find_bads_ecg)
        monkeypatch.setattr(mne.preprocessing, 'find_ecg_events',
                            lambda *a, **k: (np.array([[250, 0, 1]]), 6, 60))

    dataset = {'raw': raw, 'subject': 'ordinary_eeg'}
    userargs = {
        'outdir': tmp_path,
        'n_components': 2,
        'zoom_window': 30,
    }
    # Exercise the library default as well as the explicit EEG-fMRI override.
    if ecg_threshold != 'auto':
        userargs['ecg_threshold'] = ecg_threshold
    result = ica_module.manual_ica(dataset, userargs)

    assert result['ica'].n_components_ == 2
    for page_name in ('single_ic.html', 'between_ic.html'):
        page = (tmp_path / 'ordinary_eeg' / page_name).read_text()
        match = re.search(r'const scores\s*=\s*(\[.*?\]);', page)
        assert match is not None
        scores = json.loads(match.group(1))
        assert len(scores) == 2
        assert all('ga_local' not in score and 'ga_flag' not in score
                   for score in scores)
        if with_aux:
            assert scores[0]['ecg_thr'] == expected_ecg_threshold
            assert scores[0]['ecg_flag']
            assert not scores[1]['ecg_flag']
            assert all(e['thr'] == 0.35 and e['flag'] for e in scores[0]['eog'])
            assert all(not e['flag'] for e in scores[1]['eog'])
            assert 'const flags       = [true, false];' in page


@pytest.mark.parametrize('threshold', [-0.1, 3.0, np.nan])
def test_manual_ica_rejects_invalid_correlation_threshold(threshold):
    ica_module = importlib.import_module('osl_ephys.preprocessing.manual_ica.ica')
    with pytest.raises(ValueError, match='absolute correlation threshold'):
        ica_module.manual_ica({}, {'eog_threshold': threshold})
