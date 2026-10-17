"""Config loading, step lookup and the manifest."""

import pytest

from osl_ephys.pipeline import config as cfg
from osl_ephys.pipeline import manifest, registry


def example(dataset, userargs):
    """A custom function with the same name as a registered step."""
    return dataset


def test_registry_names_and_sections():
    def run_osl_example(dataset, userargs):
        return dataset

    registry.register(run_osl_example, "preproc")
    assert registry.find_function("example", "preproc").func is run_osl_example
    assert registry.find_function("run_osl_example", "preproc").func is run_osl_example
    assert registry.section_of("example") == "preproc"

    # A custom function wins over a registered one, and a bare MNE method works
    assert (
        registry.find_function("example", "preproc", extra_funcs=[example]).func
        is example
    )
    assert registry.find_function("resample", "preproc").name == "resample"
    with pytest.raises(LookupError, match="No step named 'nonsense'"):
        registry.find_function("nonsense", "preproc")


def test_load_config_normalises_steps_and_meta():
    config = cfg.load_config("""
        preproc:
          - filter: {l_freq: 1, h_freq: 45}
          - resample
          - bad_segments: null
        group:
          - fix_sign_ambiguity: {}
        """)
    assert cfg.steps(config, "preproc") == [
        ("filter", {"l_freq": 1, "h_freq": 45}),
        ("resample", {}),
        ("bad_segments", {}),
    ]
    assert cfg.steps(config, "group") == [("fix_sign_ambiguity", {})]
    assert cfg.steps(config, "coreg") == []
    assert config["meta"]["event_codes"] is None
    assert list(config) == ["preproc", "group", "meta"]


def test_load_config_rejects_bad_input():
    with pytest.raises(KeyError, match="Unknown config section"):
        cfg.load_config({"preprocessing": []})
    with pytest.raises(KeyError, match="no steps"):
        cfg.load_config({"preproc": None})
    with pytest.raises(ValueError, match="single key"):
        cfg.load_config({"preproc": [{"filter": {}, "resample": {}}]})
    with pytest.raises(KeyError, match="event_codes"):
        cfg.load_config({"preproc": [{"find_events": {}}]})


def nonsense(dataset, userargs):
    return dataset


def test_validate_finds_unknown_steps():
    config = cfg.load_config(
        {"preproc": [{"resample": {"sfreq": 100}}, {"nonsense": {}}]}
    )
    with pytest.raises(LookupError, match="nonsense"):
        cfg.validate(config)
    cfg.validate(config, extra_funcs=[nonsense])
    cfg.validate(cfg.load_config({"preproc": [{"resample": {"sfreq": 100}}]}))


def test_manifest_from_rows_csv_and_files(tmp_path):
    rows = manifest.load_manifest(
        [
            {"id": "s1", "subject": "sub-01", "raw_file": "a.fif"},
            {"id": "s2", "subject": "sub-01", "raw_file": "b.fif"},
        ]
    )
    assert manifest.columns(rows) == {"id", "subject", "raw_file"}
    assert list(manifest.by_subject(rows)) == ["sub-01"]

    csv_file = tmp_path / "sessions.csv"
    csv_file.write_text("id,raw_file,smri_file\ns1,a.fif,\ns2,b.fif,b_T1w.nii.gz\n")
    rows = manifest.load_manifest(csv_file)
    assert "smri_file" not in rows[0] and rows[1]["smri_file"] == "b_T1w.nii.gz"

    rows = manifest.load_manifest(["/data/sub-01_task-rest_meg.fif"])
    assert rows[0]["id"] == "sub-01_task-rest_meg"

    with pytest.raises(ValueError, match="unique"):
        manifest.load_manifest([{"id": "s1"}, {"id": "s1"}])


def test_output_paths_follow_the_layout(tmp_path):
    paths = manifest.output_paths({"id": "s1", "subject": "sub-01"}, tmp_path)
    assert paths["preproc_file"] == tmp_path / "preproc" / "s1_preproc-raw.fif"
    assert paths["surfaces_dir"] == tmp_path / "surfaces" / "sub-01"
    assert paths["session_dir"] == tmp_path / "s1"
    paths = manifest.output_paths(
        {"id": "s1", "preproc_file": "/elsewhere/s1.fif"}, tmp_path
    )
    assert str(paths["preproc_file"]) == "/elsewhere/s1.fif"
