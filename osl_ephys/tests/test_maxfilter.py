"""The MaxFilter commands, built without running MaxFilter."""

from osl_ephys import files
from osl_ephys.maxfilter import maxfilter, output_file
from osl_ephys.maxfilter.steps import maxfilter as maxfilter_step
from osl_ephys.pipeline import manifest


def test_the_command_has_the_flags():
    options = {
        "maxpath": "maxfilter",
        "tsss": True,
        "st": 10,
        "corr": 0.98,
        "movecomp": True,
        "headpos": True,
        "bads": "2233 2312",
        "scanner": "Neo",
        "autobad": False,
    }
    command = maxfilter._build_command(options, "in.fif", "out/in_proc-tsss.fif")
    assert command.startswith("maxfilter -f in.fif -o out/in_proc-tsss.fif")
    for flag in [
        "-st 10 -corr 0.98",
        "-movecomp",
        "-hp out/in_proc-tsss_headpos.log",
        "-bad 2233 2312",
        "-autobad off",
        "-cal",
        "-ctc",
    ]:
        assert flag in command, flag


def test_output_names_follow_bids():
    assert (
        output_file("sub-01_task-rest_meg.fif", "out")
        == "out/sub-01_task-rest_proc-sss_meg.fif"
    )
    assert (
        output_file("sub-01_task-rest_meg.fif", "out", tsss=True)
        == "out/sub-01_task-rest_proc-tsss_meg.fif"
    )
    assert (
        output_file("x.fif", "out", mode="multistage", tsss=True, trans="default")
        == "out/x_proc-tsss+trans.fif"
    )
    assert output_file("x.fif", "out", mode="cbu") == "out/x_proc-tsss+trans.fif"


def test_the_step_in_a_dry_run(tmp_path, capsys):
    session = {"id": "elekta", "raw_file": str(files.test_data.elekta)}
    dataset = {"session": session, "paths": manifest.output_paths(session, tmp_path)}
    dataset = maxfilter_step(
        dataset, {"mode": "multistage", "tsss": True, "dryrun": True}
    )
    assert dataset["maxfilter_file"].startswith(str(tmp_path / "maxfilter"))
    assert dataset["maxfilter_file"].endswith("_proc-tsss.fif")
    printed = capsys.readouterr().out
    assert "-autobad on" in printed and "-st 10" in printed
