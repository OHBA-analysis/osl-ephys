"""The QC report: one table across sessions, next to the plots of the selected
session.

Each section saves its numbers in a ``summary.json`` and its plots as WebP
files under the QC directory (``qc_dir/<section>/<id>/``, see
:func:`osl_ephys.preproc.qc.save_qc_plots` and
:func:`osl_ephys.parcellation.plotting.save_qc_plots`), and the report is
built from whatever has been saved so far. Sort the table by a metric to see the worst
sessions first. Only the plots of the selected session are loaded, so the
report works for datasets with tens of thousands of sessions. The report and
its plots are all in the QC directory, which can be moved or served on its
own.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# The plots of each tab, relative to the QC directory, which has a directory
# for each step holding a directory for each session, subject or head model.
# Plots that have not been saved are not shown.
TABS = {
    "Preprocessing": [
        "preproc/{id}/flowchart.webp",
        "preproc/{id}/sum_square.webp",
        "preproc/{id}/sum_square_exclude_bads.webp",
        "preproc/{id}/channel_stds.webp",
        "preproc/{id}/sensors.webp",
    ],
    "Spectra": [
        "preproc/{id}/psd.webp",
        "preproc/{id}/psd_full.webp",
        "preproc/{id}/frequency_bands.webp",
    ],
    "Artefacts": [
        "preproc/{id}/ica.webp",
        "preproc/{id}/eog.webp",
        "preproc/{id}/ecg.webp",
        "preproc/{id}/head_movement.webp",
    ],
    "Acquisition": [
        "preproc/{id}/raw_data.webp",
        "preproc/{id}/digitisation.webp",
        "preproc/{id}/events.webp",
        "preproc/{id}/evoked.webp",
    ],
    "Surfaces": [
        "surfaces/{subject}/surfaces.webp",
    ],
    "MNI Registration": [
        "surfaces/{subject}/mni_registration.webp",
    ],
    "Coregistration": [
        "coreg/{head_model}/coreg.webp",
    ],
    "Parcellation": [
        "parc/{id}/psd_topo.webp",
        "parc/{id}/power_maps.webp",
    ],
    "Sign flipping": [
        "sign_flip/sign_flip.webp",
    ],
}

# Notes shown under the plots of a tab
CAPTIONS = {
    "Preprocessing": (
        "Bad segments are shaded red on the sum-square time series; the "
        "dashed lines on the channel distributions are the bad channels."
    ),
    "Surfaces": (
        "Yellow: brain surface, cyan: inner skull, magenta: scalp. Each should "
        "follow its boundary and lie inside the next."
    ),
    "MNI Registration": (
        "Red: edges of the MNI152 template. They should follow the anatomy."
    ),
    "Sign flipping": (
        "Correlation of each session's embedded covariance with the template "
        "session's, before and after flipping."
    ),
}

# The columns of the table, read from each step's summary.json
METRICS = [
    "Bad segments (%)",
    "Bad mag (%)",
    "Bad grad (%)",
    "Bad EEG (%)",
    "Bad channels",
    "ICA excluded",
    "EOG events",
    "ECG events",
    "Head movement (mm)",
    "Epochs",
    "Epochs dropped",
    "MNI registration",
    "MNI overlap",
    "MNI MI",
    "Coreg rms (mm)",
    "Min dipoles per parcel",
    "Sign flip corr",
]

# The manifest columns the pipeline reads; any other column of a session is
# shown in the table as it is (e.g. notes taken at the scanner)
MANIFEST_COLUMNS = {
    "id",
    "subject",
    "head_model_id",
    "raw_file",
    "smri_file",
    "preproc_file",
    "epoch_file",
    "pos_file",
    "elc_file",
    "localizer_file",
}

CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    margin: 0;
    height: 100vh;
    display: flex;
    flex-direction: column;
    background: #f5f5f5;
    color: #333;
    font-size: 13px;
}
header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 16px;
    background: #fff;
    border-bottom: 1px solid #ddd;
}
header h1 {
    font-size: 18px;
    margin: 0;
}
header input[type=text] {
    padding: 5px 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-family: monospace;
    width: 220px;
}
header .info {
    color: #888;
    margin-left: auto;
}
main {
    flex: 1;
    display: flex;
    min-height: 0;
}
#table-pane {
    max-width: 60%;
    display: flex;
    flex-direction: column;
    background: #fff;
    border-right: 1px solid #ddd;
}
#table-scroll {
    flex: 1;
    overflow: auto;
}
table {
    border-collapse: collapse;
    width: 100%;
}
th, td {
    padding: 3px 7px;
    text-align: right;
    white-space: nowrap;
}
th:first-child, td:first-child {
    text-align: left;
    font-family: monospace;
}
th {
    position: sticky;
    top: 0;
    background: #eee;
    cursor: pointer;
    user-select: none;
}
th .count {
    display: block;
    color: #999;
    font-weight: normal;
    font-size: 11px;
}
tbody tr {
    cursor: pointer;
    border-bottom: 1px solid #f0f0f0;
}
tbody tr:hover {
    background: #f5f9ff;
}
tbody tr.selected {
    background: #dbe9ff;
}
td.flag {
    color: #c62828;
    font-weight: bold;
}
td.missing {
    color: #ccc;
}
#pager {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 6px;
    border-top: 1px solid #ddd;
    color: #888;
}
button {
    padding: 4px 12px;
    border: 1px solid #ccc;
    background: #fff;
    border-radius: 4px;
    cursor: pointer;
}
button:hover {
    background: #e8e8e8;
}
#plot-pane {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
}
#tabs {
    display: flex;
    gap: 4px;
    padding: 8px 12px 0 12px;
    border-bottom: 1px solid #ddd;
}
#tabs button {
    border-radius: 6px 6px 0 0;
    border-bottom: none;
    background: #e0e0e0;
}
#tabs button.active {
    background: #fff;
    font-weight: bold;
}
#plots {
    flex: 1;
    overflow: auto;
    padding: 12px;
    background: #fff;
}
#plots h2 {
    font-size: 14px;
    font-family: monospace;
    margin: 0 0 10px 0;
}
#plots img {
    max-width: 100%;
    display: block;
    margin: 0 auto 10px auto;
}
.caption {
    color: #666;
    text-align: center;
    margin: 0;
}
.placeholder {
    background: #eee;
    color: #999;
    padding: 40px;
    text-align: center;
    border-radius: 4px;
    font-style: italic;
}
"""

JS = """
const columns = DATA.columns, rows = DATA.rows, tabs = Object.keys(DATA.tabs);
const nInfo = 3;  // id, subject and head model come before the metrics
const pageSize = 200;
let order = [], selected = 0, tab = 0, sortColumn = null, ascending = true;

// Values far from the rest of a column are highlighted: more than 5 scaled
// median absolute deviations from the median (and anything that is true)
const medians = {}, mads = {};
for (let c = nInfo; c < columns.length; c++) {
    const x = rows.map(r => r[c]).filter(v => typeof v === 'number');
    if (x.length === 0) continue;
    medians[c] = median(x);
    mads[c] = 1.4826 * median(x.map(v => Math.abs(v - medians[c])));
}

function median(x) {
    const s = [...x].sort((a, b) => a - b), m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

function flagged(row, c) {
    const v = row[c];
    if (v === true) return true;
    if (typeof v !== 'number' || !mads[c]) return false;
    return Math.abs(v - medians[c]) / mads[c] > 5;
}

function format(v) {
    if (v === null) return '-';
    if (typeof v === 'number') return String(parseFloat(v.toPrecision(3)));
    return String(v);
}

function update() {
    const text = document.getElementById('filter').value.toLowerCase();
    const onlyFlagged = document.getElementById('only-flagged').checked;
    order = [];
    rows.forEach((row, i) => {
        if (text && !row[0].toLowerCase().includes(text)) return;
        if (onlyFlagged && !columns.some((_, c) => c >= nInfo && flagged(row, c))) return;
        order.push(i);
    });
    if (sortColumn !== null) {
        order.sort((a, b) => {
            const x = rows[a][sortColumn], y = rows[b][sortColumn];
            if (x === y) return a - b;
            if (x === null) return 1;
            if (y === null) return -1;
            return (x < y ? -1 : 1) * (ascending ? 1 : -1);
        });
    }
    selected = 0;
    render();
}

function sortBy(c) {
    ascending = sortColumn === c ? !ascending : true;
    sortColumn = c;
    update();
}

function render() {
    // Table, one page at a time
    const page = Math.floor(selected / pageSize);
    const nPages = Math.max(1, Math.ceil(order.length / pageSize));
    let html = '<thead><tr>';
    columns.forEach((name, c) => {
        if (c > 0 && c < nInfo) return;
        const n = rows.filter(r => r[c] !== null).length;
        const arrow = sortColumn === c ? (ascending ? ' &#9650;' : ' &#9660;') : '';
        html += `<th onclick="sortBy(${c})">${name}${arrow}<span class="count">${n}</span></th>`;
    });
    html += '</tr></thead><tbody>';
    for (let i = page * pageSize; i < Math.min(order.length, (page + 1) * pageSize); i++) {
        const row = rows[order[i]];
        html += `<tr id="row-${i}" class="${i === selected ? 'selected' : ''}" onclick="select(${i})">`;
        row.forEach((v, c) => {
            if (c > 0 && c < nInfo) return;
            const cls = v === null ? 'missing' : (c >= nInfo && flagged(row, c) ? 'flag' : '');
            html += `<td class="${cls}">${format(v)}</td>`;
        });
        html += '</tr>';
    }
    document.getElementById('table').innerHTML = html + '</tbody>';
    document.getElementById('page').textContent = `Page ${page + 1} / ${nPages}`;
    document.getElementById('count').textContent = `${order.length} / ${rows.length} sessions`;
    const tr = document.getElementById(`row-${selected}`);
    if (tr) tr.scrollIntoView({block: 'nearest'});

    // Tabs
    document.getElementById('tabs').innerHTML = tabs.map((name, t) =>
        `<button class="${t === tab ? 'active' : ''}" onclick="showTab(${t})">${name}</button>`
    ).join('');

    // Plots of the selected session
    const plots = document.getElementById('plots');
    if (order.length === 0) {
        plots.innerHTML = '<div class="placeholder">No sessions</div>';
        return;
    }
    const [id, subject, headModel] = rows[order[selected]];
    const path = file => file.replace('{id}', id).replace('{subject}', subject)
        .replace('{head_model}', headModel);
    plots.innerHTML = `<h2>${id}</h2>` + DATA.tabs[tabs[tab]].map(file =>
        `<img src="${path(file)}" onerror="missing(this)">`
    ).join('') + `<p class="caption">${DATA.captions[tabs[tab]] || ''}</p>`
        + '<div class="placeholder" id="placeholder" hidden>Not available</div>';
}

function missing(img) {
    img.remove();
    if (!document.querySelector('#plots img')) {
        document.getElementById('placeholder').hidden = false;
    }
}

function select(i) {
    if (i < 0 || i >= order.length) return;
    selected = i;
    render();
}

function showTab(t) {
    tab = (t + tabs.length) % tabs.length;
    render();
}

document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' && e.target.type === 'text') return;
    if (e.key === 'ArrowDown') select(selected + 1);
    else if (e.key === 'ArrowUp') select(selected - 1);
    else if (e.key === 'ArrowRight') showTab(tab + 1);
    else if (e.key === 'ArrowLeft') showTab(tab - 1);
    else return;
    e.preventDefault();
});

update();
"""


@contextmanager
def headless():
    """Draw with the Agg backend, so saving QC plots never opens a window."""
    backend = plt.get_backend()
    plt.switch_backend("Agg")
    try:
        yield
    finally:
        plt.close("all")
        plt.switch_backend(backend)


def save_figure(fig, filename: str | Path, close: bool = True) -> None:
    """Save a QC plot the way the report expects it.

    WebP files are several times smaller than PNG and look the same. The plots
    are about 1000 pixels wide, the width the report shows them at. Any other
    extension is saved with the same settings minus the WebP quality.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure.
    filename : str or Path
        Where to save it; the extension picks the format.
    close : bool, optional
        Close the figure after saving.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"dpi": 100, "bbox_inches": "tight"}
    if filename.suffix.lower() in (".webp", ".jpg", ".jpeg"):
        kwargs["pil_kwargs"] = {"quality": 80}
    fig.savefig(filename, **kwargs)
    if close:
        plt.close(fig)


def save_summary(directory: str | Path, summary: dict) -> None:
    """Write a step's numbers to ``directory/summary.json`` for the report."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "summary.json", "w") as file:
        json.dump(summary, file, indent=2)


def _read_json(path: Path) -> dict | None:
    """Read a JSON file, or return None if it has not been saved."""
    try:
        with open(path) as file:
            return json.load(file)
    except (OSError, ValueError):
        return None


def _coreg_dir(outdir: Path, head_model: str) -> Path | None:
    """The directory holding a head model's coregistration, on either route."""
    for name in ("coreg", "freesurfer"):
        if (outdir / head_model / name / "coreg.json").exists():
            return outdir / head_model / name
    return None


def _session_row(
    session: dict,
    qc_dir: Path,
    outdir: Path | None,
    registrations: dict,
    sign_flip: dict | None,
) -> dict:
    """The report's row for a session: the numbers its steps saved.

    Parameters
    ----------
    session : dict
        The session's manifest row: ``id``, and ``subject`` and
        ``head_model_id`` if it has them.
    qc_dir : Path
        The QC directory.
    outdir : Path, optional
        The output directory, where the surfaces and coregistration keep their
        numbers and plots.
    registrations : dict
        MNI registration quality of the subjects read so far, shared by the
        sessions of a subject.
    sign_flip : dict, optional
        The sign flipping summary, by session id.

    Returns
    -------
    row : dict
        Session id, subject, the id owning the coregistration and the metrics
        that have been saved.
    """
    id_ = session["id"]
    subject = session.get("subject", id_)
    head_model = session.get("head_model_id", id_)
    row = {"Session": id_, "subject": subject, "head_model": head_model}

    for key, value in session.items():
        if key not in MANIFEST_COLUMNS and isinstance(value, (str, int, float, bool)):
            row[key] = value

    summary = _read_json(qc_dir / "preproc" / id_ / "summary.json")
    if summary is not None:
        for key, column in [
            ("bad_percent", "Bad segments (%)"),
            ("bad_percent_mag", "Bad mag (%)"),
            ("bad_percent_grad", "Bad grad (%)"),
            ("bad_percent_eeg", "Bad EEG (%)"),
            ("n_bad_channels", "Bad channels"),
            ("ica_n_excluded", "ICA excluded"),
            ("n_eog_events", "EOG events"),
            ("n_ecg_events", "ECG events"),
            ("head_movement_mm", "Head movement (mm)"),
            ("n_epochs", "Epochs"),
            ("n_epochs_dropped", "Epochs dropped"),
        ]:
            if key in summary:
                row[column] = summary[key]

    summary = _read_json(qc_dir / "parc" / id_ / "summary.json")
    if summary is not None and "min_dipoles_per_parcel" in summary:
        row["Min dipoles per parcel"] = summary["min_dipoles_per_parcel"]

    if sign_flip is not None and id_ in sign_flip:
        row["Sign flip corr"] = sign_flip[id_]["after"]

    if outdir is None:
        return row

    if subject not in registrations:
        registrations[subject] = _read_json(
            outdir / "surfaces" / subject / "mni_registration.json"
        )
    registration = registrations[subject]
    if registration is not None:
        used = registration["used"]
        row["MNI registration"] = used
        row["MNI overlap"] = registration[used]["dice"]
        row["MNI MI"] = registration[used]["mutual_information"]

    coreg_dir = _coreg_dir(outdir, head_model)
    if coreg_dir is not None:
        row["Coreg rms (mm)"] = _read_json(coreg_dir / "coreg.json").get("rms")
    return row


def _copy_plots(table: pd.DataFrame, qc_dir: Path, outdir: Path) -> None:
    """Copy the surface and coregistration plots into the QC directory.

    Surface extraction and coregistration save their plots with their output
    (the preprocessing and parcellation plots are saved in the QC directory).
    The surfaces are copied once per subject and the coregistration once per
    head model, to the paths in TABS, as WebP files. Plots that have not
    changed since they were last copied are skipped.
    """
    copies = []
    for subject in table["subject"].dropna().unique():
        for name in ["surfaces", "mni_registration"]:
            source = outdir / "surfaces" / subject / f"{name}.png"
            copies.append((source, qc_dir / "surfaces" / subject / f"{name}.webp"))
    for head_model in table["head_model"].unique():
        coreg_dir = _coreg_dir(outdir, head_model)
        if coreg_dir is not None:
            source = coreg_dir / "coreg.png"
            copies.append((source, qc_dir / "coreg" / head_model / "coreg.webp"))

    def copy(files):
        source, destination = files
        if not source.exists():
            return
        if (
            destination.exists()
            and destination.stat().st_mtime >= source.stat().st_mtime
        ):
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Moved into place when complete, so an interrupted copy is redone
        temporary = destination.with_suffix(".tmp")
        Image.open(source).save(temporary, "WEBP", quality=80)
        temporary.replace(destination)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(copy, copies))


def _discovered_plots(qc_dir: Path, pattern: str) -> list[str]:
    """The preprocessing plots matching a pattern that any session saved.

    For the plots whose names are not fixed: the custom figures and the
    further ICA plots of a session with several ICA runs.
    """
    preproc = qc_dir / "preproc"
    names = (
        sorted({p.name for p in preproc.glob(f"*/{pattern}")})
        if preproc.is_dir()
        else []
    )
    return [f"preproc/{{id}}/{name}" for name in names]


def generate_report(
    qc_dir: str | Path,
    sessions: list[dict],
    outdir: str | Path | None = None,
    metrics: pd.DataFrame | None = None,
    output_file: str = "report.html",
) -> Path:
    """Generate the QC report.

    Builds a table with one row per session from the QC files the pipeline
    steps have saved so far, shown next to the QC plots of the selected
    session. The table can be sorted by each metric and filtered, and values
    far from the rest of their column are highlighted. Sessions are navigated
    with the up/down arrows and the plots with the left/right arrows.

    Parameters
    ----------
    qc_dir : str or Path
        The QC directory. The preprocessing, parcellation and sign flipping
        QC is read from here and the report is written here.
    sessions : list of dict
        The sessions (the manifest): each has an ``id``, and a ``subject``
        and ``head_model_id`` when they differ from it.
    outdir : str or Path, optional
        The output directory. If given, the surface extraction and
        coregistration QC is read from here and their plots are copied to the
        QC directory.
    metrics : pd.DataFrame, optional
        More columns for the table (e.g. statistics of the parcellated data),
        indexed by session id. Numbers and booleans are highlighted like the
        other metrics.
    output_file : str, optional
        Filename for the report, written in ``qc_dir``.

    Returns
    -------
    path : Path
        The report.
    """
    qc_dir = Path(qc_dir)
    if outdir is not None:
        outdir = Path(outdir)
    sign_flip = _read_json(qc_dir / "sign_flip" / "summary.json")

    # One row per session, keeping the metrics that at least one session has.
    # The QC files are read in threads, a large dataset has tens of thousands
    registrations = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(
            pool.map(
                lambda s: _session_row(s, qc_dir, outdir, registrations, sign_flip),
                sessions,
            )
        )
    info = ["Session", "subject", "head_model"]
    table = pd.DataFrame(rows)
    table = table.drop(columns=[c for c in table.columns if table[c].isna().all()])
    # The metrics in a fixed order, then the manifest's own columns
    columns = [c for c in METRICS if c in table.columns]
    columns += [c for c in table.columns if c not in info and c not in METRICS]
    table = table[info + columns]
    if outdir is not None:
        _copy_plots(table, qc_dir, outdir)
    if metrics is not None:
        table = table.join(metrics, on="Session")

    tabs = {name: list(plots) for name, plots in TABS.items()}
    tabs["Artefacts"] += _discovered_plots(qc_dir, "ica_*.webp")
    custom = _discovered_plots(qc_dir, "custom_*.webp")
    if custom:
        tabs["Custom"] = custom

    data = json.loads(table.to_json(orient="split", double_precision=6))
    data = {
        "columns": data["columns"],
        "rows": data["data"],
        "tabs": tabs,
        "captions": CAPTIONS,
    }
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    data_json = json.dumps(data).replace("</", "<\\/")
    report = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>QC Report</title>
<style>{CSS}</style>
</head>
<body>
<header>
<h1>QC Report</h1>
<input type="text" id="filter" placeholder="Filter sessions..." oninput="update()">
<label><input type="checkbox" id="only-flagged" onchange="update()"> highlighted only</label>
<span id="count"></span>
<span class="info">&#8593;&#8595; sessions &nbsp; &#8592;&#8594; plots &nbsp; | &nbsp; Generated: {timestamp}</span>
</header>
<main>
<div id="table-pane">
<div id="table-scroll"><table id="table"></table></div>
<div id="pager">
<button onclick="select((Math.floor(selected / pageSize) - 1) * pageSize)">&#9664;</button>
<span id="page"></span>
<button onclick="select((Math.floor(selected / pageSize) + 1) * pageSize)">&#9654;</button>
</div>
</div>
<div id="plot-pane">
<div id="tabs"></div>
<div id="plots"></div>
</div>
</main>
<script>const DATA = {data_json};{JS}</script>
</body>
</html>"""

    output_path = qc_dir / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"Report saved: {output_path}")
    return output_path
