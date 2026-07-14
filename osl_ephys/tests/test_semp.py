"""Unit tests for the semp (Simultaneous EEG-fMRI Preprocessing) subpackage.

Covers the wrappers and helpers refactored in 2026-07: the mode-based
``create_epoch`` family (incl. the forgotten/spurious trigger-spacing check),
``slice_reject`` (reuses the fitted ICA), the rolling timer, ``ensure_dir``,
the ``_amplitude_yticks`` extraction, the single-source metric helpers,
``proc_userargs`` / ``require_keys``, and AAS/OBS smoke/regression checks.

The heavy stages (real EEG-fMRI ICA) are not exercised; everything runs on
small synthetic ``mne`` objects so the suite is fast and deterministic.
"""
import numpy as np
import mne
import pytest

mne.set_log_level("ERROR")

from osl_ephys.preprocessing.batch import find_func
from osl_ephys.preprocessing.semp.wrappers.epoching import (
    create_epoch, create_TR_epoch, create_He_epoch, simulate_epoch,
    _check_fixed_spacing)
from osl_ephys.preprocessing.semp.wrappers.ica import slice_reject
from osl_ephys.preprocessing.semp.wrappers.aas import epoch_aas
from osl_ephys.preprocessing.semp.wrappers.obs import epoch_obs
from osl_ephys.preprocessing.semp.wrappers import timer as timer_mod
from osl_ephys.preprocessing.semp.vis import _amplitude_yticks
from osl_ephys.preprocessing.semp.utils import (
    ensure_dir, proc_userargs, require_keys, mne_epoch2raw)
import osl_ephys.preprocessing.semp as semp


# ------------------------------------------------------------------ fixtures --

def _synthetic_raw(T=30.0, sfreq=100.0):
    """30 s of noise @100 Hz with R128 (2 s) and HE (1 s) trigger annotations."""
    raw = mne.io.RawArray(
        np.random.RandomState(0).randn(2, int(T * sfreq)) * 1e-5,
        mne.create_info(["C1", "C2"], sfreq, "eeg"), verbose="ERROR")
    tr = np.arange(1.0, T - 2.5, 2.0)
    he = np.arange(0.5, T - 1.5, 1.0)
    raw.set_annotations(mne.Annotations(
        np.r_[tr, he], 0.0, ["R128"] * len(tr) + ["HE"] * len(he)))
    return raw


def _dataset():
    return {"raw": _synthetic_raw(), "tr_interval": 2.0,
            "tr_event_key": ["R128"], "he_event_key": ["HE"]}


# --------------------------------------------------------------- registration -

@pytest.mark.parametrize("name", [
    "create_epoch", "create_TR_epoch", "create_He_epoch", "simulate_epoch",
    "slice_reject", "epoch_aas", "crop_TR",
    "manual_ica", "apply_ica"])   # these two live in preprocessing.manual_ica
def test_wrappers_resolve_by_name(name):
    assert find_func(name).__name__ == f"run_osl_{name}"


def test_removed_names_do_not_resolve():
    # renamed / relocated in the 2026-07 cleanup
    assert find_func("slice_ica") is None
    assert find_func("set_channel_type_raw") is None


# ---------------------------------------------------------------- epoching ----

def test_create_TR_epoch_fixed_window():
    d = create_TR_epoch(_dataset(), {})
    ep = d["tr_ep"]
    assert len(ep) == 14 and np.isclose(ep.tmax, 2.0)


def test_create_epoch_fixed_matches_TR_wrapper():
    d = create_epoch(_dataset(), {"mode": "fixed", "event_key": ["R128"],
                                  "tmin": 0, "tmax": 2.0, "epoch_key": "tr_ep"})
    assert len(d["tr_ep"]) == 14


def test_create_epoch_requires_epoch_key():
    with pytest.raises(ValueError):
        create_epoch(_dataset(), {"mode": "fixed", "event_key": ["R128"],
                                  "tmin": 0, "tmax": 2.0})     # epoch_key now required


def test_create_He_epoch_auto_window():
    d = create_He_epoch(_dataset(), {})
    ep = d["he_ep"]
    # window auto-sized from the ~1 s trigger spacing (capped at the max gap)
    assert len(ep) == 28 and 0.9 < ep.tmax <= 1.03 * 1.0 + 1e-9


def test_simulate_epoch_grid_and_jitter():
    d = simulate_epoch(_dataset(), {"width": 1.0})
    assert "sim_ep" in d and np.isclose(d["sim_ep"].tmax, 1.0)
    # jitter runs; edge onsets may be dropped, so allow a small count delta
    d2 = simulate_epoch(_dataset(), {"width": 1.0, "jitter": 0.2})
    assert abs(len(d2["sim_ep"]) - len(d["sim_ep"])) <= 2
    with pytest.raises(ValueError):
        simulate_epoch(_dataset(), {})            # width required


def test_create_epoch_rejects_old_event_and_simulate_mode():
    with pytest.raises(KeyError):
        create_epoch(_dataset(), {"event": "TR"})          # old switch gone
    with pytest.raises(ValueError):
        create_epoch(_dataset(), {"mode": "simulate", "tmax": 1.0})
    with pytest.raises(ValueError):
        create_epoch(_dataset(), {"mode": "fixed"})        # needs event_key


def test_create_epoch_random_uses_given_key():
    # random surrogate now stores under the provided epoch_key (no '_rand' suffix)
    d = create_epoch(_dataset(), {"mode": "fixed", "event_key": ["R128"],
                                  "tmin": 0, "tmax": 2.0, "epoch_key": "tr_ep",
                                  "random": True})
    assert "tr_ep" in d and "tr_ep_rand" not in d


# ------------------------------------------------------ fixed-spacing check ---

def test_check_fixed_spacing_regular_is_ok():
    onsets = np.arange(20) * 100.0                    # perfectly regular
    f = _check_fixed_spacing(onsets, sfreq=100.0)
    assert f["status"] == "ok" and f["n_missing"] == 0


def test_check_fixed_spacing_detects_forgotten_trigger():
    onsets = np.delete(np.arange(20) * 100.0, 5)      # drop one -> a 2x gap
    f = _check_fixed_spacing(onsets, sfreq=100.0)
    assert f["status"] == "grid" and f["n_missing"] == 1


def test_check_fixed_spacing_small_jitter_not_reported():
    rng = np.random.RandomState(0)
    onsets = np.cumsum(100.0 + rng.uniform(-3, 3, size=25))   # <=3% jitter
    f = _check_fixed_spacing(onsets, sfreq=100.0)
    assert f["status"] == "ok" and f["n_missing"] == 0


def test_check_fixed_spacing_irregular_warns():
    rng = np.random.RandomState(1)
    onsets = np.sort(rng.choice(np.arange(5000), size=25, replace=False)).astype(float)
    f = _check_fixed_spacing(onsets, sfreq=100.0)
    assert f["status"] == "irregular"


def _capture_log_or_print(func, capsys):
    """Return a semp wrapper's ``log_or_print`` output regardless of routing.

    ``log_or_print`` *prints* when no osl logger is configured (this file run
    alone) but routes to ``osl_logger.info`` once the ``osl_ephys`` logger is
    set up -- which an earlier test in the *full* suite does, and that logger
    has ``propagate: false``, so the message never reaches stdout/``capsys``.
    Grab both: a handler on the ``osl_ephys`` logger for the log path, plus
    ``capsys`` for the print path.
    """
    import logging
    logger = logging.getLogger("osl_ephys")
    grabbed = []

    class _Grab(logging.Handler):
        def emit(self, record):
            grabbed.append(record.getMessage())

    handler = _Grab()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        func()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    return capsys.readouterr().out + "\n".join(grabbed)


def test_create_TR_epoch_warns_on_forgotten_trigger(capsys):
    raw = mne.io.RawArray(np.zeros((1, 2000)),
                          mne.create_info(["C1"], 100.0, "eeg"), verbose="ERROR")
    onsets = np.delete(np.arange(1.0, 19.0, 1.0), 5)    # 1 s TRs, one missing
    raw.set_annotations(mne.Annotations(onsets, 0.0, ["R128"] * len(onsets)))
    ds = {"raw": raw, "tr_interval": 1.0, "tr_event_key": ["R128"]}
    out = _capture_log_or_print(
        lambda: create_TR_epoch(ds, {"correct_trig": False}), capsys)
    assert "FORGOTTEN" in out
    # opting out is silent
    raw.set_annotations(mne.Annotations(onsets, 0.0, ["R128"] * len(onsets)))
    ds = {"raw": raw, "tr_interval": 1.0, "tr_event_key": ["R128"]}
    out = _capture_log_or_print(
        lambda: create_TR_epoch(ds, {"correct_trig": False, "check_spacing": False}), capsys)
    assert "FORGOTTEN" not in out


# ---------------------------------------------------------------- slice_reject -

def _multichan_raw(n_ch=6, T=20.0, sfreq=100.0):
    raw = mne.io.RawArray(
        np.random.RandomState(2).randn(n_ch, int(T * sfreq)) * 1e-5,
        mne.create_info([f"E{i}" for i in range(n_ch)], sfreq, "eeg"), verbose="ERROR")
    return raw


def _fit_ica(raw, n=4):
    ica = mne.preprocessing.ICA(n_components=n, max_iter=200, random_state=0)
    ica.fit(raw, verbose="ERROR")
    return ica


def test_slice_reject_reuses_ica_and_unions_exclude():
    raw = _multichan_raw()
    ica = _fit_ica(raw)
    ica.exclude = [1]                       # pretend ica_autoreject flagged IC1
    ds = {"raw": raw.copy(), "ica": ica, "slice_interval": 0.055, "tr_interval": 2.1}
    out = slice_reject(ds, {})
    assert out["ica"] is ica                # no second ICA fit
    assert 1 in out["ica"].exclude          # EOG/ECG marks preserved (union)


def test_slice_reject_requires_fitted_ica():
    with pytest.raises(KeyError):
        slice_reject({"raw": _synthetic_raw(), "slice_interval": 0.055,
                      "tr_interval": 2.1}, {})


# --------------------------------------------------------------------- AAS ----

def test_epoch_aas_removes_a_stationary_template():
    """A perfectly repeated per-volume artefact should be almost fully removed
    by average artefact subtraction (identical epochs -> windowed mean ~ epoch)."""
    sfreq, TR, n_vol = 100.0, 1.0, 24
    tmpl = np.sin(2 * np.pi * 12 * np.arange(int(TR * sfreq)) / sfreq) * 1e-4
    sig = np.tile(tmpl, n_vol) + np.random.RandomState(1).randn(n_vol * len(tmpl)) * 1e-7
    raw = mne.io.RawArray(np.stack([sig, sig]),
                          mne.create_info(["C1", "C2"], sfreq, "eeg"), verbose="ERROR")
    onsets = np.arange(n_vol) * TR
    raw.set_annotations(mne.Annotations(onsets, 0.0, ["R128"] * n_vol))
    ds = {"raw": raw, "tr_interval": TR, "tr_event_key": ["R128"]}
    ds = create_TR_epoch(ds, {"correct_trig": False})
    pre_var = float(np.var(ds["raw"].get_data()))
    ds = epoch_aas(ds, {"epoch_key": "tr_ep", "picks": "eeg",
                        "window_length": 6, "fit": False})
    post_var = float(np.var(ds["raw"].get_data()))
    # a perfectly repeated per-volume template is almost entirely subtracted;
    # variance collapses by >10x (residual is only the epoch-overlap edges).
    assert post_var < 0.1 * pre_var


# -------------------------------------------------------------------- timer ---

def test_timer_start_while_running_warns_not_raises():
    reg = timer_mod._TimerRegistry()
    reg.clear_history("t")
    reg._running.pop("t", None)
    reg.start("t")
    reg.start("t")                          # stale still running -> must NOT raise
    elapsed = reg.end("t")
    assert elapsed >= 0.0
    # rolling group history accumulates across "subjects"
    reg.start("t"); reg.end("t")
    assert len(reg.get_history("t")) == 2


# ---------------------------------------------------------------- ensure_dir --

def test_ensure_dir_created_flag_and_idempotent(tmp_path):
    d = tmp_path / "a" / "b" / "c"
    assert ensure_dir(d) is True            # created
    assert ensure_dir(d) is False           # already exists, no raise
    assert d.is_dir()


# ------------------------------------------------------------------- metric ---

def test_mean_psd_in_band_single_source_and_fallback():
    # metric primitives live once in semp.metric; the package surfaces them too
    assert semp.mean_psd_in_band is semp.metric.mean_psd_in_band
    assert semp.EEGTracer is semp.metric.EEGTracer
    freqs = np.array([1.0, 2.0, 3.0, 4.0])
    row = np.array([10.0, 20.0, 30.0, 40.0])
    assert semp.mean_psd_in_band(row, freqs, 2.5, 1.0) == 25.0   # mean of bins 2,3
    # window narrower than the resolution -> nearest bin
    assert semp.mean_psd_in_band(row, freqs, 2.4, 0.01) == 20.0


# ---------------------------------------------------------------------- vis ----

def test_unpack_plot_handles_both_return_shapes():
    from osl_ephys.preprocessing.semp.vis import _unpack_plot

    class _Fig:                     # MNE fig-only return
        axes = ["ax0", "ax1"]
    fig = _Fig()
    assert _unpack_plot(fig) == (fig, fig.axes)         # fig-only -> (fig, fig.axes)
    assert _unpack_plot((fig, "AXES")) == (fig, "AXES")  # (fig, axes) passthrough


def test_psd_and_temp_plot_smoke(tmp_path):
    """psd_plot / temp_plot after the helper extraction: exercise Raw + numpy
    inputs and the single/list/none event-raster branches, saving to disk."""
    import matplotlib
    matplotlib.use("Agg")
    from osl_ephys.preprocessing.semp.vis import psd_plot, temp_plot

    raw = _synthetic_raw(T=20.0, sfreq=200.0)
    data = raw.get_data()
    n = data.shape[1]
    onsets = np.arange(200, n - 200, 200).astype(int)
    events = np.column_stack([onsets, np.zeros_like(onsets), np.ones_like(onsets)])

    psd_plot(raw, name="s", save_pth=tmp_path / "psd", fmax=50)
    assert (tmp_path / "psd.pdf").exists()
    psd_plot(data, name="np", fs=200.0, save_pth=tmp_path / "psd_np", fmax=50)
    assert (tmp_path / "psd_np.pdf").exists()

    temp_plot(raw, 0, length=n, events=events, event_id=1,
              save_pth=str(tmp_path / "t1.png"))                       # single series
    temp_plot(raw, 0, length=n, events=[events, events], event_id=[1, 1],
              event_name=["a", "b"], save_pth=str(tmp_path / "t2.png"))  # list of series
    temp_plot(data, 0, fs=200.0, save_pth=str(tmp_path / "t3.png"))    # no events
    assert all((tmp_path / f).exists() for f in ("t1.png", "t2.png", "t3.png"))


# ------------------------------------------------------------- amplitude ticks -

def test_amplitude_yticks_style():
    seg = np.linspace(-3e-5, 3e-5, 50)
    vals, labels, ylim, exp = _amplitude_yticks(seg)
    assert exp == -5                                  # 3e-5 -> factor 1e-5
    assert all(len(lbl.split(".")[1]) == 2 for lbl in labels)   # 2-decimal labels
    assert len(vals) in (7, 9, 11)
    assert ylim[0] <= seg.min() and ylim[1] >= seg.max()


def test_amplitude_yticks_degenerate_zero_signal():
    vals, labels, ylim, exp = _amplitude_yticks(np.zeros(10))
    assert len(vals) == len(labels) and ylim[0] < ylim[1]


# ------------------------------------------------- mne_epoch2raw seam modes ---

def _overlap_setup(n_ep, spacing=50, L=100, base=1000, sfreq=100.0):
    """A 1-ch raw (filled with -1) + `n_ep` epochs of length L, onsets `spacing`
    apart (so consecutive epochs overlap by L-spacing). Epoch i's data is the
    constant value (i + 1). Returns (raw, epochs, data-ndarray)."""
    raw = mne.io.RawArray(np.full((1, base), -1.0),
                          mne.create_info(["C1"], sfreq, "eeg"), verbose="ERROR")
    onsets = 100 + spacing * np.arange(n_ep)
    events = np.column_stack([onsets, np.zeros(n_ep, int), np.ones(n_ep, int)]).astype(int)
    epochs = mne.Epochs(raw.copy(), events, tmin=0, tmax=(L - 1) / sfreq,
                        baseline=None, preload=True, verbose="ERROR")
    assert len(epochs) == n_ep
    data = np.stack([np.full((1, L), float(i + 1)) for i in range(n_ep)])
    return raw, epochs, data


def test_mne_epoch2raw_new_overwrites_at_seam():
    # two epochs: [100,200)=1 and [150,250)=2; 'new' -> later wins the overlap
    raw, epochs, data = _overlap_setup(2)
    d = mne_epoch2raw(epochs, raw.copy(), data, tmin=0, overwrite="new").get_data()[0]
    assert d[149] == 1.0 and d[150] == 2.0 and d[199] == 2.0


def test_mne_epoch2raw_even_seam_at_midpoint():
    # 'even' keeps the datapoint closer to its event onset -> seam at the
    # midpoint between the two epoch centres (samples 151 and 201 -> 177)
    raw, epochs, data = _overlap_setup(2)
    d = mne_epoch2raw(epochs, raw.copy(), data, tmin=0, overwrite="even").get_data()[0]
    assert d[176] == 1.0 and d[177] == 2.0


def test_mne_epoch2raw_obs_switches_new_to_even_after_11():
    # 'obs' = 'new' for the first 11 epochs (i<=10), 'even' after. With 13
    # epochs: the early seam is new-style, the late one is even-style.
    raw, epochs, data = _overlap_setup(13)
    d = mne_epoch2raw(epochs, raw.copy(), data, tmin=0, overwrite="obs").get_data()[0]
    # early overlap (epochs 0/1): 'new' -> epoch1 wins from its onset (150)
    assert d[150] == 2.0
    # late overlap (epochs 11/12, i=12>10): 'even' -> epoch11 still shows at 700,
    # because epoch12 is trimmed to start at 727 ('new' would give 13 here)
    assert d[700] == 12.0
    # sanity: plain 'new' *would* give 13 at that sample
    dn = mne_epoch2raw(epochs, raw.copy(), data, tmin=0, overwrite="new").get_data()[0]
    assert dn[700] == 13.0


# --------------------------------------------------------------------- OBS ----

# ----------------------------------------------- proc_userargs / require_keys -

def test_proc_userargs_strict_rejects_unknown_key():
    assert proc_userargs({"a": 2}, {"a": 1, "b": 0}) == {"a": 2, "b": 0}
    with pytest.raises(KeyError):
        proc_userargs({"typo": 1}, {"a": 1})            # unexpected key
    # strict=False keeps the old silent-passthrough behaviour
    assert proc_userargs({"typo": 1}, {"a": 1}, strict=False)["typo"] == 1


def test_wrappers_reject_unknown_userarg():
    # a typo'd key must fail loudly, not silently no-op a batch
    with pytest.raises(KeyError):
        create_TR_epoch(_dataset(), {"windo_length": 6})
    with pytest.raises(KeyError):
        epoch_aas(_dataset(), {"winddow_length": 6})


def test_require_keys_reports_missing():
    require_keys({"a": 1}, "a", "stage")                # present -> no raise
    require_keys({"a": 1}, ["a"], "stage")
    with pytest.raises(KeyError) as e:
        require_keys({"a": None, "b": 2}, ["a", "c"], "mystage")
    msg = str(e.value)
    assert "mystage" in msg and "'a'" in msg and "'c'" in msg  # None counts as missing


def test_epoch_obs_removes_low_rank_artifact():
    """Identical per-volume epochs are rank-1 across epochs, so an OBS pass with
    npc=1 models them fully and the epoched region collapses to ~0."""
    sfreq, TR, n_vol = 100.0, 1.0, 24
    tmpl = np.sin(2 * np.pi * 10 * np.arange(int(TR * sfreq)) / sfreq) * 1e-4
    sig = np.tile(tmpl, n_vol)                       # every volume identical
    raw = mne.io.RawArray(np.stack([sig, sig]),
                          mne.create_info(["C1", "C2"], sfreq, "eeg"), verbose="ERROR")
    raw.set_annotations(mne.Annotations(np.arange(n_vol) * TR, 0.0, ["R128"] * n_vol))
    ds = {"raw": raw, "tr_interval": TR, "tr_event_key": ["R128"]}
    ds = create_TR_epoch(ds, {"correct_trig": False})
    pre_var = float(np.var(ds["tr_ep"].get_data()))
    ds = epoch_obs(ds, {"epoch_key": "tr_ep", "picks": "eeg", "npc": 1,
                        "remove_mean": True, "overwrite": "even"})
    post_var = float(np.var(ds["raw"].get_data()))
    assert post_var < 0.05 * pre_var
    # the removed-noise raw is stashed for the report
    assert isinstance(ds["noise_tr_ep"], mne.io.RawArray)
