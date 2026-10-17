"""Every subcommand of the osl-ephys command parses its arguments."""

import pytest

from osl_ephys.pipeline.cli import main


@pytest.mark.parametrize(
    "command",
    [
        [],
        ["run"],
        ["maxfilter"],
        ["ica"],
        ["ica", "review"],
        ["ica", "apply"],
        ["func"],
        ["download-data"],
        ["check"],
    ],
)
def test_help(command, capsys):
    with pytest.raises(SystemExit) as exit:
        main(command + ["--help"])
    assert exit.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_a_missing_subcommand_is_an_error():
    with pytest.raises(SystemExit) as exit:
        main([])
    assert exit.value.code != 0
