"""The FSL route of the surface extraction, on the MNI152 brain (needs FSL)."""

import json

import pytest

from osl_ephys import files, rhino


@pytest.mark.fsl
def test_extract_surfaces_on_the_mni152_brain(tmp_path):
    smri = files.mask.file("MNI152_T1_2mm_brain.nii.gz")
    outdir = tmp_path / "surfaces"
    rhino.extract_surfaces(smri, str(outdir), include_nose=False)
    assert (outdir / "surfaces.png").exists()
    quality = json.loads((outdir / "mni_registration.json").read_text())
    assert quality[quality["used"]]["dice"] > 0.9
