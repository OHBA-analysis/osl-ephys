"""Finding the step a config names."""

import functools

from osl_ephys.pipeline import registry
from osl_ephys.preproc import steps


def test_steps_of_the_preproc_section():
    assert registry.find_function("notch_filter", "preproc").func is steps.notch_filter
    assert registry.find_function("resample", "preproc").func is steps.resample
    assert registry.find_function("bad_segments", "preproc").func is steps.bad_segments
    assert registry.find_function("bad_channels", "preproc").func is steps.bad_channels
    assert registry.find_function("ica_label", "preproc").func is steps.ica_label


def test_mne_methods_are_steps():
    for name in ["crop", "savgol_filter", "filter"]:
        found = registry.find_function(name, "preproc").func
        assert isinstance(found, functools.partial)
        assert found.func is steps.mne_method
        assert found.keywords["method"] == name


def test_custom_functions_come_first():
    def filter(dataset, userargs):
        return dataset

    assert (
        registry.find_function("filter", "preproc", extra_funcs=[filter]).func is filter
    )


def test_semp_steps_are_registered_lazily():
    assert registry.find_function("epoch_aas", "preproc").func.__name__ == "epoch_aas"
