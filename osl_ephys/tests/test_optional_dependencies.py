"""osl-ephys imports without the packages only the GLM module needs."""

import subprocess
import sys


def test_import_without_glmtools_and_sails():
    code = (
        "import sys; sys.modules['glmtools'] = None; sys.modules['sails'] = None; "
        "import osl_ephys; from osl_ephys import glm, utils; "
        "print(osl_ephys.__version__)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]


def test_using_them_gives_a_clear_error():
    code = (
        "import sys; sys.modules['glmtools'] = None; "
        "from osl_ephys.glm import glm_base; glm_base.glm.fit"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "pip install glmtools" in result.stderr
