"""Coregistration, forward model, beamforming and parcellation on the test
recordings, against the MNI152 surfaces (no FSL needed)."""

import mne
import pytest

from osl_ephys import files
from osl_ephys.pipeline import run_batch

CONFIG = """
preproc:
  - resample: {sfreq: 100}
coreg:
  - extract_fiducials_and_headshape: {}
  - coregister: {use_nose: false, allow_mri_scaling: true}
source_recon:
  - forward_model: {model: Single Layer, gridstep: 12}
parcellation:
  - beamform_and_parcellate:
      chantypes: [mag, grad]
      rank: {meg: 60}
      parcellation_file: atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz
      orthogonalisation: symmetric
group:
  - sign_flip: {n_embeddings: 5, method: spectral}
"""


@pytest.fixture(scope="module")
def sessions():
    return [
        {"id": "elekta", "subject": "sub-01", "raw_file": files.test_data.elekta},
        {"id": "elekta2", "subject": "sub-01", "raw_file": files.test_data.elekta},
    ]


@pytest.mark.fsl
def test_source_reconstruction_on_mni152_surfaces(sessions, tmp_path):
    results = run_batch(CONFIG, sessions, tmp_path, n_workers=2)
    assert {s for _, s, _, _ in results} == {"ok"}, results

    for id_ in ("elekta", "elekta2"):
        coreg = tmp_path / id_ / "coreg"
        assert (coreg / "head_scaledmri-trans.fif").exists()
        assert (coreg / "coreg.png").exists()
        assert (coreg / "model-fwd.fif").exists()
        assert (tmp_path / id_ / "src" / "filters-lcmv.h5").exists()
        parc = mne.io.read_raw_fif(
            tmp_path / id_ / f"{id_}_parc-raw.fif", verbose=False
        )
        assert len(mne.pick_types(parc.info, misc=True)) == 52
        assert (tmp_path / id_ / f"{id_}_sflip_parc-raw.fif").exists()
    assert (tmp_path / "qc" / "parc" / "elekta" / "psd_topo.webp").exists()
    assert (tmp_path / "qc" / "coreg" / "elekta" / "coreg.webp").exists()
    assert (tmp_path / "qc" / "sign_flip" / "sign_flip.webp").exists()
    report = (tmp_path / "qc" / "report.html").read_text()
    assert "Coreg rms (mm)" in report and "Sign flip corr" in report
    assert "Min dipoles per parcel" in report


MINIMUM_NORM_CONFIG = """
preproc:
  - resample: {sfreq: 100}
coreg:
  - extract_fiducials_and_headshape: {}
  - coregister: {use_nose: false, allow_mri_scaling: true}
source_recon:
  - forward_model: {model: Single Layer, gridstep: 12}
  - minimum_norm: {chantypes: [mag, grad], rank: {meg: 60}}
parcellation:
  - parcellate:
      source_method: minimum_norm
      parcellation_file: atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz
      orthogonalisation: symmetric
"""


@pytest.mark.fsl
def test_volumetric_minimum_norm_on_mni152_surfaces(sessions, tmp_path):
    results = run_batch(
        MINIMUM_NORM_CONFIG, sessions[:1], tmp_path, n_workers=1, qc_dir=False
    )
    assert results[0][1] == "ok", results
    assert (tmp_path / "elekta" / "freesurfer" / "op-inv.fif").exists()
    parc = mne.io.read_raw_fif(
        tmp_path / "elekta" / "elekta_parc-raw.fif", verbose=False
    )
    assert len(mne.pick_types(parc.info, misc=True)) == 52
