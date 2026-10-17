"""The parallel runner: logs, failure isolation and the summary."""

import csv
import time
from pathlib import Path

from osl_ephys.pipeline import runner


def process(item):
    """A session that succeeds, warns or fails depending on its id."""
    print(f"verbose output for {item['id']}")
    if item["id"] == "slow":
        time.sleep(0.2)
    elif item["id"] == "partial":
        runner.warning("no ICA components were rejected")
    elif item["id"] == "broken":
        raise ValueError("cannot read the raw file")
    runner.log("done")


def test_runner_isolates_failures_and_summarises(tmp_path):
    items = [{"id": "slow"}, {"id": "partial"}, {"id": "broken"}, {"id": "fine"}]
    log_dir = tmp_path / "logs"
    reported = []

    results = runner.run(
        process,
        items,
        n_workers=2,
        log_dir=log_dir,
        report=lambda i: reported.append(len(i)),
    )

    status = {id_: s for id_, s, _, _ in results}
    assert status == {
        "slow": "ok",
        "partial": "partial",
        "broken": "failed",
        "fine": "ok",
    }
    assert reported == [4]

    # A log per item; the failed one is renamed and holds the error
    assert (log_dir / "slow.log").exists()
    assert (log_dir / "fine.log").read_text().count("verbose output for fine") == 1
    assert not (log_dir / "broken.log").exists()
    assert "cannot read the raw file" in (log_dir / "broken.log.err").read_text()

    with open(log_dir / "summary.csv") as f:
        rows = {r["id"]: r for r in csv.DictReader(f)}
    assert rows["partial"]["status"] == "partial"
    assert "no ICA components" in rows["partial"]["message"]
    assert rows["broken"]["message"].startswith("ValueError")
