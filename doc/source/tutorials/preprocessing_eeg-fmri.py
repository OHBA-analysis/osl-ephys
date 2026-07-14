'''
Preprocessing simultaneous EEG-fMRI with semp
=============================================

The preprocessing tutorials so far have dealt with "clean-room" MEG/EEG: data recorded outside a scanner, where the main artefacts are physiological (blinks, heartbeat, muscle) plus the odd bad channel. **Simultaneous EEG-fMRI** is a different game. The EEG is recorded *inside* a running MR scanner, so on top of the usual artefacts it carries two enormous, structured artefacts that are orders of magnitude larger than the brain signal:

1. **The gradient artefact (GA)** --- induced every time the scanner switches its field gradients to acquire a slice. It is periodic, locked to the volume repetition time (TR) and to the within-volume slice timing, and can be ~100x the EEG amplitude.
2. **The ballistocardiogram (BCG / pulse artefact)** --- induced by tiny head/electrode movements with every heartbeat in the static field. Roughly locked to the cardiac cycle, but more variable than the GA.

``osl_ephys.preprocessing.semp`` ("**S**imultaneous **E**EG-f**M**RI **P**reprocessing") is a subpackage that adds the wrappers needed to remove these artefacts and slots them into the same ``run_proc_chain`` / ``run_proc_batch`` machinery you already know. If you have done the :doc:`preprocessing_manual`, :doc:`preprocessing_automatic` and :doc:`preprocessing_batch` tutorials, you already understand most of how to drive it.

Unlike the other tutorials (which use the Wakeman & Henson dataset --- recorded *outside* a scanner, so it has no gradient or pulse artefact), this one uses a **real, openly available simultaneous EEG-fMRI dataset**: `NATVIEW_EEGFMRI <https://github.com/NathanKlineInstitute/NATVIEW_EEGFMRI/tree/main>`_ from the Nathan Kline Institute, hosted on AWS S3. We will walk it end-to-end, in the order you would actually do it:

0. **Get the data**
1. **The semp subpackage: stages resolved by name**
2. **The artefact-removal strategy: epoch, average, subtract**
3. **Step 1 --- Locate your files: a pathfinder**
4. **Step 2 --- Discover your acquisition metadata** (TR, slice timing, trigger, mains)
5. **Step 3 --- The** ``initialize`` **extra_func**
6. **Step 4 --- The preprocessing config, stage by stage**
7. **Running the batch and reading the diagnostics**
8. **Concluding remarks**

.. note::
   Every concrete number, channel name and trigger label below was read off the real NATVIEW files. Swap them for your own dataset's values where indicated. The heavy stages (loading 5 kHz ``.set`` files, ICA on EEG-fMRI) are not run inside this rendered page, but the code is exactly what you would execute.

'''

#%%
# Step 0 --- Get the data
# ^^^^^^^^^^^^^^^^^^^^^^^
# NATVIEW lives in a public, requester-free S3 bucket. Grab it with the AWS CLI (the ``--no-sign-request`` flag means no AWS account is needed):
#
# .. code-block:: bash
#
#     pip install awscli
#
#     # everything (large -- many subjects x sessions x tasks):
#     aws s3 sync s3://fcp-indi/data/Projects/NATVIEW_EEGFMRI/raw_data/ \
#         /path/to/natview/raw_data --no-sign-request
#
# The full dataset is big, so for this tutorial pull a **subset** --- a couple of subjects is plenty to build and check a pipeline:
#
# .. code-block:: bash
#
#     aws s3 sync s3://fcp-indi/data/Projects/NATVIEW_EEGFMRI/raw_data/ \
#         /path/to/natview/raw_data --no-sign-request \
#         --exclude "*" --include "sub-01/*" --include "sub-05/*"
#
# Pick one ``base_path`` for the project; we keep the raw download and the output we are about to create side by side under it. In this tutorial:

BASE = "/path/to/natview"      # <- your base_path
RAW  = f"{BASE}/raw_data"      # the S3 download
OUT  = f"{BASE}/semp_output"   # where this pipeline writes

#%%
# **What's in it.** NATVIEW is BIDS-organised: ``sub-XX/ses-XX/{eeg,func,anat}/``. The EEG for each task comes as both BrainVision (``.vhdr``/``.vmrk``) and EEGLAB (``.set``); the S3 download ships the ``.set`` (data inline) and the ``.vhdr``/``.vmrk`` header+markers, but **not** the BrainVision ``.eeg`` binary --- so read the ``.set`` (osl-ephys / MNE handle EEGLAB natively). Each subject did several tasks (``rest``, ``checker``, ``inscapes``, movie clips, ...). We use rest as an example.

#%%
# The semp subpackage: stages resolved by name
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Recall from the batch tutorial that osl-ephys resolves a stage name in a config (e.g. ``{'filter': {...}}``) through ``find_func``, which looks, in order: at any ``extra_funcs`` you passed, then at the built-in ``run_osl_<name>`` wrappers, then ``run_mne_<name>``, then plain MNE ``Raw`` / ``Epochs`` methods.
#
# Every semp wrapper (``crop_TR``, ``create_epoch``, ``epoch_aas``, ``epoch_obs``, ``slice_reject``, ...) is registered with ``find_func`` as a ``run_osl_<name>`` built-in --- exactly like ``manual_ica`` (see the :doc:`preprocessing_manual-ica` tutorial). So you reference any semp stage in a config *by name*, using the plain osl-ephys runner, with no special import and no ``extra_funcs`` entry:

from osl_ephys.preprocessing import run_proc_batch, run_proc_chain

#%%
# (For backward compatibility ``from osl_ephys.preprocessing.semp import run_proc_batch`` still works --- it is now literally the same function.) You still pass your own *project* functions --- like the ``initialize`` extra_func below --- via ``extra_funcs=``; only the semp library wrappers are resolved automatically.
#
# The semp wrappers fall into a few groups:
#
# - **Cropping / epoching**: ``crop_TR`` (trim to the fMRI acquisition window), ``create_epoch`` (cut trigger-locked epochs; ``create_TR_epoch`` / ``create_He_epoch`` are the easy per-TR / per-helium-pump variants, and ``simulate_epoch`` lays a triggerless grid), ``crop_by_epoch``, ``mid_crop``.
# - **Artefact templates**: ``epoch_aas`` (average artefact subtraction --- the GA), ``epoch_obs`` (optimal basis sets --- the BCG), ``epoch_ssp``.
# - **ICA**: ``slice_reject`` (flag + reject the residual slice-harmonic components of the ICA fitted by ``ica_raw``), ``manual_ica`` (browser review --- see the :doc:`preprocessing_manual-ica` tutorial), ``apply_ica``.
# - **Housekeeping / QA**: ``voltage_correction``, ``cleanup``, and the diagnostic ``init_tracer`` / ``ckpt_report`` / ``summary`` trio.
#
# .. note::
#    semp uses a few dependencies beyond core osl-ephys (``pandas`` / ``seaborn`` / ``nibabel`` / ``nilearn``, mostly for its plotting and source-recon helpers; the artefact-template maths is plain ``numpy``). The ``run_osl_<name>`` registration imports semp **lazily**, only when a semp stage actually runs, so core ``osl_ephys.preprocessing`` keeps a minimal dependency surface: a vanilla install without those extras still imports and runs non-semp pipelines fine, and you only need them once a semp stage executes.

#%%
# The artefact-removal strategy: epoch, average, subtract
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The GA and the BCG are both (quasi-)periodic, and that is exactly what semp exploits. The recipe for both is the same three moves:
#
# 1. **Epoch** the continuous raw data on the artefact's own clock --- one epoch per scanner volume (TR) for the GA, one epoch per heartbeat for the BCG. This is what ``create_epoch`` does; the resulting ``mne.Epochs`` is stashed back into the ``dataset`` dict under a key like ``'tr_ep'``.
# 2. **Estimate an artefact template** from those epochs. For the GA, the template is a *local sliding average* across neighbouring volumes (average artefact subtraction, AAS --- ``epoch_aas``): because brain activity is not time-locked to the gradients but the artefact is, averaging a window of volumes cancels the brain signal and leaves the artefact. For the BCG, the template is the leading principal components of the heartbeat epochs (optimal basis sets, OBS --- ``epoch_obs``).
# 3. **Subtract** the template from every epoch and stitch the cleaned epochs back into the continuous ``dataset['raw']``.
#
# A few semp-specific details worth knowing:
#
# - ``epoch_aas`` slides a window of ``window_length`` volumes and subtracts the windowed mean. With ``fit=False`` it subtracts the raw average template; with ``fit=True`` it least-squares-fits the template's amplitude per channel before subtracting (useful when the artefact amplitude drifts). ``pre_pad`` controls how the window is centred at the start/end of the recording.
# - ``epoch_obs`` keeps ``npc`` principal components (default 3). ``remove_mean`` should be ``True`` for TR- or heartbeat-length epochs and ``False`` for very short slice-length epochs (removing the mean of a sub-0.1 s epoch would eat real signal). The ``screen_high_power`` / ``pc_from_spurious`` knobs let you keep residual-GA or motion epochs out of the PC estimate so the BCG template is not contaminated.
# - Both wrappers stash the artefact template (``pc_<epoch_key>``) and the removed noise (``noise_<epoch_key>``) back into the ``dataset`` so the diagnostic ``ckpt_report`` can plot exactly what was taken out.
#
# After template subtraction there is usually a residual *slice* artefact at high harmonics of the slice-timing frequency (because AAS assumes a perfectly stationary template, which never quite holds). ``slice_reject`` mops this up: rather than fit its own ICA, it **reuses the one fitted by** ``ica_raw`` in the general-ICA stage (below), measures each component's power at the slice harmonics relative to a baseline band, and adds the components whose ratio exceeds ``noise2base_threshold`` to ``ica.exclude`` (alongside the EOG/ECG ones). It needs ``dataset['slice_interval']`` and ``dataset['tr_interval']`` set --- which brings us to the practical part.

#%%
# Step 1 --- Locate your files: a pathfinder
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# An EEG-fMRI study has many files per recording scattered across folders. Hard-coding paths gets unmanageable fast, so semp pipelines drive file lookup through a **pathfinder**: you declare a template per *kind* of file with named placeholders, and it maps a compact recording id to the path of any kind. NATVIEW is already consistently named (BIDS), so no renaming is needed --- we just describe the layout.
#
# The modern pathfinder is ``osl_pathfinder.Pathfinder`` (``pip install osl-pathfinder``): template-based, no subclassing. Give it a ``paths`` dict (kind -> template), an ``id`` template saying how the placeholder fields concatenate into a compact id, and an ``anchor`` kind that is globbed to discover which recordings exist on disk:

from osl_pathfinder import Pathfinder

pf = Pathfinder(
    paths={
        # anchor: the input EEG (.set) --- here the rest recording. We only use
        # one task, so the kind is just "raw" rather than "rest" (the outputs
        # below would otherwise be "rest_preproc"/"rest_afterica" --- redundant
        # when there is nothing else to disambiguate from). {subject:02d}/
        # {session:02d} pad to the BIDS "sub-01"/"ses-01" form and consume the
        # pad on parse, so ids round-trip both ways.
        "raw":      f"{RAW}/sub-{{subject:02d}}/ses-{{session:02d}}/eeg/"
                    f"sub-{{subject:02d}}_ses-{{session:02d}}_task-rest_eeg.set",
        # per-recording outputs we will create:
        "preproc":  f"{OUT}/{{subject}}{{session}}/{{subject}}{{session}}_preproc-raw.fif",
        "afterica": f"{OUT}/{{subject}}{{session}}/{{subject}}{{session}}_after_ica-raw.fif",
    },
    id="{subject:d}{session:1d}",   # sub-01 ses-01 -> "11"; sub-10 ses-02 -> "102"
    anchor="raw",
)

#%%
# That gives you a small, explicit API used throughout a semp project:
#
# - ``pf.ids`` --- every recording id discovered by globbing the anchor on disk (e.g. ``'11'`` for sub-01/ses-01).
# - ``pf.id2path('preproc', file_id)`` --- the path of any kind for a recording.
# - ``pf.path2id('raw', some_path)`` --- the inverse: recover the id from a filename (used to set ``dataset['subject']``).
# - ``pf.exists('raw', file_id)`` --- check a kind is on disk before using it.
# - ``pf.id2field(file_id)`` --- the integer fields (``{'subject': 1, 'session': 1}``), handy when you must glob something the templates don't cover (e.g. the ``anat/`` T1w for later source recon).

#%%
# **Scaling up: one pathfinder for a whole study, with** ``derive``. Above we used a *single-task* pathfinder (only ``task-rest``), so the id only had to encode subject and session. A larger study has many task types per recording --- NATVIEW has a dozen: the run-less ``rest``, ``checker``, ``checkeroff``, ``checkerout``, ``inscapes``, ``peer``, and the run-paired ``dme``, ``dmh``, ``tp``, ``monkey1``, ``monkey2``, ``monkey5`` (each recorded as both ``_run-01`` *and* ``_run-02``) --- and you have a choice. Keeping one small pathfinder per task (``rest_pf``, ``monkey1_pf``, ...) is perfectly fine. Or you can drive everything from *one* pathfinder whose id also encodes the task (and run).
#
# You probably do not want a verbose id like ``'01-01-dme-02'`` though --- you want a compact code. ``Pathfinder`` supports that with ``derive``: an ``f(fields, kind) -> extra_fields`` hook that fills in a placeholder *computed* from the id (and, going the other way, recovers the id's code from what is on disk). The trick is to fold the variable part of the filename --- the task token *and* its optional ``_run-NN`` --- into a single derived ``stem`` field, so the path template needs no explicit ``task`` or ``run`` placeholder at all. The id then reads as ``<subject><session><task><run>``, where the task is a short **letters-only** code sitting between the leading digits and the trailing run digit --- so it splits cleanly at the digit/letter boundary, with no separators and no zero-padding. For example,
#
# ::
#
#     .../sub-01/ses-01/eeg/sub-01_ses-01_task-dme_run-02_eeg.set       <->  id '11d2'
#     .../sub-01/ses-01/eeg/sub-01_ses-01_task-monkey1_run-02_eeg.set   <->  id '11ka2'
#     .../sub-01/ses-01/eeg/sub-01_ses-01_task-rest_eeg.set             <->  id '11r1'
#
# How does ``derive`` know that ``dme`` files carry a ``_run-NN`` while ``rest`` files do not? It does **not** hold a hard-coded list of run-less tasks. It reads it off the disk: a given task's recordings either *all* carry ``_run-NN`` or none do, so one glob settles it. And going the other way (disk ``->`` id), a filename with no ``_run-NN`` simply **defaults its run to 1**. So the whole run-less story is "no ``_run-`` on disk => run defaults to 1", never a maintained set:

import glob
from functools import lru_cache

# one short LETTERS-ONLY code per task (your choice); the inverse recovers the
# full token. Codes are letters (monkey1 -> 'ka', not 'k1') so the id splits
# unambiguously at the digit/letter boundary -- see the note below.
ABBR = {'rest': 'r', 'checker': 'c', 'checkeroff': 'cf', 'checkerout': 'co',
        'inscapes': 'i', 'peer': 'p', 'dme': 'd', 'dmh': 'h',
        'monkey1': 'ka', 'monkey2': 'kb', 'monkey5': 'ke', 'tp': 't'}
FULL = {v: k for k, v in ABBR.items()}

@lru_cache(maxsize=None)
def _has_run(task):
    # A task's recordings either ALL carry _run-NN or none do; discover which
    # from the disk once, rather than enumerating a run-less set by hand.
    return bool(glob.glob(f"{RAW}/sub-*/ses-*/eeg/*_task-{task}_run-*_eeg.set"))

def derive(fields, kind):
    f, out = dict(fields), {}
    # disk -> id: split the stem into task code + run. A filename with no
    # _run-NN defaults its run to 1 (there is no run segment to read).
    if 'stem' in f and 'tk' not in f:
        task, _, runpart = str(f['stem']).partition('_run-')
        if task in ABBR:
            out['tk'] = ABBR[task]
            out['run'] = int(runpart) if runpart else 1
    # id -> disk: rebuild the stem, adding _run-NN only for tasks that carry
    # it on disk (discovered above -- not from a hard-coded run-less set). A
    # run-less task exists only at run 1, so any other run gets a _run-NN stem
    # that simply won't resolve -- '11r1' finds the rest file, '11r2' fails.
    if 'tk' in f and 'stem' not in f:
        task, run = FULL[str(f['tk'])], int(f['run'])
        out['stem'] = task if (not _has_run(task) and run == 1) \
            else f"{task}_run-{run:02d}"
    return out

big_pf = Pathfinder(
    paths={'raw': f"{RAW}/sub-{{subject:02d}}/ses-{{session:02d}}/eeg/"
                  f"sub-{{subject:02d}}_ses-{{session:02d}}_task-{{stem}}_eeg.set"},
    id="{subject:d}{session:1d}{tk:l}{run:1d}",   # sub-01/ses-01 dme run-02 -> '11d2'
    anchor='raw',
    derive=derive,
)

# big_pf.id2path('raw', '11d2')          # -> ...task-dme_run-02_eeg.set, for subject 1
# big_pf.id2path('raw', '111ka2')         # -> ...task-monkey1_run-02_eeg.set, for subject 11
# big_pf.id2path('raw', '211r1')          # -> ...task-rest_eeg.set   (run-less), for subject 21
# big_pf.path2id('raw', some_dme_file)   # -> id like '11d2'
# sorted(big_pf.ids)                     # all subjects x sessions x tasks, from disk

#%%
# **A word on how the id stays parseable.** The id fields are concatenated with **no separators** (``11ka2``), so ``Pathfinder`` must be able to split the string back into fields. Adjacent fields need a *decidable boundary*, and there are two kinds: a **fixed-width** field (which consumes a known number of characters), or a **character-class change** (digits give way to letters, or vice versa). This id leans on the second: ``subject`` and ``session`` are digits (``{subject:d}`` / ``{session:1d}``), the task code ``tk`` is **letters** (``{tk:l}``), and ``run`` is a digit again (``{run:1d}``). So the parser reads the leading digit-run as ``subject`` + ``session`` (subject is all but the last digit, session the last), then the letters as ``tk``, then the final digit as ``run`` --- every boundary lands on a digit/letter transition or a known width. That is why the task codes are **letters-only** (``monkey1 -> ka``, not ``k1``): a digit inside the code would blur into the subject/run digits and make the split ambiguous.
#
# .. note::
#    Two upshots of the digit/letter rule. (i) ``subject`` needs **no** zero-padding: ``{subject:d}`` is fine because the letters of ``tk`` mark exactly where the subject+session digits end --- so ``sub-1`` gives ``'11...'`` and ``sub-10`` gives ``'101...'``, both unambiguous. (ii) If you gave ``tk`` a numeric code, or put two variable-width digit fields side by side with no fixed width between them, ``Pathfinder`` would refuse to build and raise an ``ambiguous id template`` error naming the offending fields. Keep the code letters-only and every id round-trips. (This class-aware disambiguation needs ``osl-pathfinder >= 0.4.1``; earlier versions required all-but-one field in a separator-free id to be fixed-width, e.g. a zero-padded ``{subject:02d}`` with a fixed ``{run:1d}``.)
#
# ``derive`` is consulted in **both** directions: with the file ``kind`` when rendering a path (``id2path`` / ``exists``), and with ``kind=None`` when mapping between the id and its fields (``id2field`` / ``field2id``, and therefore ``path2id`` / ``scan`` / ``pf.ids``). As long as your hook is idempotent and can fill either side from the other --- here, ``stem -> (tk, run)`` and ``(tk, run) -> stem`` --- ids and paths round-trip, and ``scan`` discovers every recording on disk just as it does for the simple single-task pathfinder. (This is a recent capability; older ``osl-pathfinder`` applied ``derive`` only on the id ``->`` path direction.) Run this ``big_pf`` against a real NATVIEW download and it discovers every recording across all 12 tasks x subjects x sessions and round-trips each one. Keeping separate per-task pathfinders is still a perfectly good choice --- this is just the tool for when you want one compact id space across the whole study.

#%%
# Step 2 --- Discover your acquisition metadata
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Several semp wrappers need numbers specific to *your* scan. NATVIEW (being BIDS) hands most of them to you in JSON sidecars; the trigger label is easiest to read off the data. Do this once for a representative recording, then bake the answers into ``initialize`` (Step 3). The values printed in the comments below are the **actual NATVIEW values**.
#
# **2a. ``tr_interval`` --- the TR (seconds).** Straight out of the BOLD JSON:

import json, numpy as np

bold = json.load(open(f"{RAW}/sub-01/ses-01/func/sub-01_ses-01_task-rest_bold.json"))
tr_interval = bold["RepetitionTime"]                       # NATVIEW: 2.1 s

#%%
# **2b. ``slice_interval`` --- the time between slice acquisitions (seconds).** The naive ``TR / len(SliceTiming)`` is a **trap**: with a multiband sequence several slices are acquired *simultaneously*, so ``SliceTiming`` repeats those values and ``len()`` over-counts by the multiband factor. Always collapse to the *unique* slice onset times first, then look at the spacing between them:

st = np.array(bold["SliceTiming"])
uniq = np.sort(np.unique(st))            # collapse multiband duplicates
steps = np.diff(uniq)                    # gap between successive slice acquisitions
vals, counts = np.unique(np.round(steps, 6), return_counts=True)
slice_interval = vals[counts.argmax()]   # the mode -- robust to a little jitter
print(f"n_unique_slices={len(uniq)}  step values={dict(zip(vals, counts))}  "
      f"slice_interval~{slice_interval:.4f}s  GA~({1/slice_interval:.2f}Hz harmonics + {1/tr_interval:.2f}Hz harmonics)")

# NATVIEW rest:  38 unique slices, steps {0.055: 30, 0.0575: 7}  -> ~0.055 s, GA ~18.2 Hz
# A multiband-4 sequence (e.g. TR=1.14, 64 entries) would show only 16 UNIQUE slices, step {0.07: 15} -> 0.07 s (GA ~14.3 Hz).

#%%
# Two things this print teaches that a single number would hide:
#
# - **The slice timing is not perfectly regular.** NATVIEW's steps are mostly 0.055 s but occasionally 0.0575 s. So ``slice_interval`` is **not an exact period** --- treat ``1/slice_interval`` as a *locator* for roughly where the gradient-artefact residual harmonics sit in the spectrum (~18 Hz, 36 Hz, ...), which is all ``slice_reject`` needs (it searches a window around each harmonic, not a single bin).
# - **Why semp does volume-level AAS, not slice-level.** A slice-by-slice average artefact subtraction (as in FASTR) assumes every slice epoch has the *same length*; with the jittered timing above they don't, so per-slice templates misalign. semp instead averages whole **volume** (TR) epochs --- whose length *is* stable, set by the rock-steady ``R128`` trigger (Step 2c) --- and then mops up the residual slice harmonics with ``slice_reject``. That combination is robust to exactly this slice-timing instability.
#
# You can sanity-check the locator against the data: the GA shows up as tall, regularly spaced peaks at harmonics of ``1/slice_interval`` Hz in the PSD of an *uncleaned* raw file::
#
#     import mne
#     raw = mne.io.read_raw_eeglab(pf.id2path('raw', '11'), preload=True)
#     raw.compute_psd(picks='eeg', fmin=0, fmax=50).plot()   # peaks near 18, 36 Hz ...
#
# **2c. ``tr_event_key`` --- the volume (TR) trigger label.** The amplifier records a marker at every fMRI volume onset. Find which annotation label that is by listing the labels and looking for the one whose *inter-event interval* equals the TR with near-zero jitter:

import mne, numpy as np

raw = mne.io.read_raw_eeglab(pf.id2path('raw', '11'), preload=False, verbose='ERROR')
events, event_id = mne.events_from_annotations(raw, verbose='ERROR')
sfreq = raw.info['sfreq']                                  # NATVIEW: 5000 Hz

for label, code in event_id.items():
    t = events[events[:, 2] == code, 0]
    if len(t) < 2:
        continue
    d = np.diff(t) / sfreq
    print(f"{label!r}: n={len(t)} mean_int={d.mean():.4f}s jitter={d.std():.4f}s")
    if abs(d.mean() - tr_interval) < 0.05 and d.std() < 0.01:
        print(f"   ^^^ tr_event_key: {label!r}")
# NATVIEW prints: 'R128': n=288 mean_int=2.1000s jitter=0.0000s  <- the TR trigger
# (R128 is BrainVision's standard MR volume-trigger label; the others are
#  stimulus / 'Sync On' markers.)

#%%
# Pass ``tr_event_key`` as a *list* of candidate labels --- the wrappers use the first one present in a given recording, which is robust to the label drifting between sessions or sites.
#
# **2d. Other knobs you can read from the sidecars.**
#
# - **Mains frequency** for the notch. Two ways to be sure, and you should agree both before trusting it: (i) *provenance* --- the EEG JSON ``PowerLineFrequency`` says 60, and NATVIEW was recorded at the Nathan Kline Institute in New York, i.e. a 60 Hz country (Europe is 50 Hz); (ii) *the data* --- find the mains peak in the PSD, which doesn't rely on the sidecar being correct::
#
#       psd = raw.compute_psd(picks='eeg', fmin=45, fmax=65)
#       freqs, p = psd.freqs, psd.get_data().mean(0)
#       print("mains peak near", round(freqs[p.argmax()]))   # NATVIEW: 60
#
#   So notch ``60 120`` (the fundamental + first harmonic), *not* the European ``50 100``.
# - **Channel types**: NATVIEW has 61 EEG channels (referenced to FCz), one ECG channel named ``ECG``, and two EOG channels ``EOGL`` / ``EOGU``, no EMG. EEGLAB ``.set`` files don't carry MNE channel *types*, so we re-assign them in the config (Step 4).
# - **Electrode positions**: NATVIEW ships ``*_electrodes.tsv`` per session, but the channels use standard 10-05 names, so a standard montage (``set_montage='standard_1005'``) is the simplest route for the ICA topographies and later source recon --- no custom montage extra_func needed.
# - **``he_event_key``** (helium-pump trigger, optional, for an OBS pass on >30 Hz content): NATVIEW has no helium-pump marker, so we leave it unset.

#%%
# Step 3 --- The ``initialize`` extra_func
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# semp does not guess the Step-2 numbers --- you hand them over in a small project ``initialize`` extra_func that runs as the very first stage of the chain. It seeds ``dataset`` with everything the later wrappers read out of it:

from functools import partial
from pathlib import Path
from osl_ephys.preprocessing.semp.utils import psd_band_ratio

def initialize(dataset, userargs):
    """Populate dataset with the metadata the semp wrappers expect (NATVIEW)."""
    # --- Step 2 values ---
    dataset['tr_interval']    = userargs.get('tr_interval', 2.1)        # 2a
    dataset['slice_interval'] = userargs.get('slice_interval', 0.055)   # 2b (mode of the
    #   slice-onset steps; a *locator* for the ~18 Hz GA harmonic, not an exact period)
    dataset['tr_event_key']   = userargs.get('tr_event_key', ['R128'])  # 2c
    dataset['he_event_key']   = userargs.get('he_event_key', [])       # 2d (none here)

    # --- where per-recording output goes ---
    dataset['target_pth'] = userargs.get('target_pth', Path(OUT))

    # --- pathfinder + a stable recording id ---
    # manual_ica / apply_ica / the report stages key their output folders on
    # dataset['subject']; set it here, NEVER via userargs (a shared batch
    # userargs dict would make every recording overwrite the same folder).
    dataset['pf']      = userargs['pf']
    dataset['subject'] = dataset['pf'].path2id('raw', dataset['raw'].filenames[0])

    dataset['orig_sfreq'] = dataset['raw'].info['sfreq']

    # --- extra diagnostic tracers for ckpt_report / summary (optional; {} is
    #     fine to just use semp's defaults). Here: slice-harmonic band ratios,
    #     so the report shows the GA collapsing as the pipeline proceeds. ---
    si = dataset['slice_interval']
    dataset['tracer'] = {
        'psd_slice':  partial(psd_band_ratio, band1=[1/si - 1, 1/si + 1],
                              band2='beta', fn1=np.mean),
        'psd_2slice': partial(psd_band_ratio, band1=[2/si - 1, 2/si + 1],
                              band2=[20, 35], fn1=np.mean),
    }

    # --- per-recording acquisition quirks would live here, keyed on the id,
    #     each with a one-line "why". (None needed for this NATVIEW subset.) ---
    # if dataset['subject'] == '42':   # e.g. crop a bad tail for sub-04/ses-02
    #     dataset['raw'].crop(tmin=0, tmax=400)

    return dataset

#%%
# Note that, unlike a dataset with digitised electrodes, NATVIEW needs no custom ``set_channel_montage`` function --- a standard montage is applied as an ordinary MNE stage in the config below. So ``initialize`` is the only project extra_func we pass.

#%%
# Step 4 --- The preprocessing config, stage by stage
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here is a complete resting-state config for NATVIEW. It is an ordinary osl-ephys ``config`` dict --- a list of single-key dicts under ``'preproc'`` --- mixing built-in osl-ephys/MNE stages (``notch_filter``, ``filter``, ``resample``, ``bad_segments``, ``bad_channels``, ``set_channel_types``, ``set_montage``) with the semp wrappers. Read it top to bottom: it tells the whole story. The comments explain *why* each choice is made and where the NATVIEW-specific values came from.

target_pth = Path(OUT)

config = {
    'preproc': [

        # -- 4.1  Init, tracer, channel types, montage, notch, TR crop -------
        {'initialize': {'target_pth': target_pth, 'pf': pf}},   # Step 3
        {'init_tracer': {}},
        # EEGLAB .set carries no channel types: name NATVIEW's ECG/EOG channels.
        {'set_channel_types': {'ECG': 'ecg', 'EOGL': 'eog', 'EOGU': 'eog'}},
        # standard montage (NATVIEW uses 10-05 names; on_missing ignores ECG/EOG):
        {'set_montage': {'montage': 'standard_1005', 'on_missing': 'ignore'}},
        # mains notch -- NATVIEW PowerLineFrequency is 60 Hz (US), so 60 + 120:
        {'notch_filter': {'freqs': '60 120'}},
        # trim to whole TR intervals using the R128 volume trigger:
        {'crop_TR': {}},   # trims to whole TRs; TR + trigger come from dataset (tr_interval / tr_event_key)
        {'ckpt_report': {'ckpt_name': 'raw', 'focus_range': [0, 10], 'dB': False}},

        # -- 4.2  Gradient artefact removal (AAS) ---------------------------
        # epoch one window per fMRI volume, then subtract a 30-volume
        # sliding-average template. create_TR_epoch is the easy variant of
        # create_epoch: fixed window = one TR (dataset['tr_interval']), trigger
        # = dataset['tr_event_key'], correct_trig (pearson-align) on by default.
        {'create_TR_epoch': {}},
        {'epoch_aas': {'epoch_key': 'tr_ep', 'overwrite': 'new',
                       'picks': 'all', 'window_length': 30, 'fit': False}},
        {'ckpt_report': {'ckpt_name': 'after_aas_removal',
                         'key_to_print': 'tr_ep', 'dB': False}},

        # -- 4.3  Band-pass + edge-crop + resample --------------------------
        # Use an IIR Butterworth, NOT FIR: FIR does not fully attenuate the
        # huge out-of-band GA residual here. Band-pass before resampling.
        # NATVIEW samples at 5000 Hz, so 125 Hz is well below Nyquist.
        {'filter': {'l_freq': 0.5, 'h_freq': 125, 'method': 'iir',
                    'iir_params': {'order': 5, 'ftype': 'butter'}}},
        # drop 5 s from each edge to remove filter ring-up before downsampling:
        {'mid_crop': {'edge': 5}},
        {'resample': {'sfreq': 250}},
        {'ckpt_report': {'ckpt_name': 'after_filt', 'dB': False}},

        # -- 4.4  Automatic bad-segment / bad-channel detection -------------
        {'bad_segments': {'segment_len': 500, 'picks': 'eeg',
                          'significance_level': 0.1, 'detect_zeros': False}},
        {'bad_segments': {'segment_len': 500, 'picks': 'eeg', 'mode': 'diff',
                          'significance_level': 0.1, 'detect_zeros': False}},
        {'bad_channels': {'picks': 'eeg', 'significance_level': 0.1}},
        {'bad_segments': {'segment_len': 2500, 'picks': 'eog', 'detect_zeros': False}},

        # -- 4.5  General ICA -- one fit, three rejections ----------------
        # A SINGLE ICA (ica_raw) serves both the pulse/ocular/cardiac cleanup
        # and the residual slice-artefact cleanup:
        #   * ica_autoreject marks EOG (correlation, EOGL/EOGU) + ECG (CTPS,
        #     ECG) components -- apply=False, so it only *marks*.
        #   * slice_reject reuses that same fitted ICA, adds the components with
        #     high power at the slice-timing harmonics (~18 Hz + multiples;
        #     residual because AAS leaves a stationary-template remnant), and
        #     applies the union once. Reads slice_interval / tr_interval from
        #     the dataset (set in initialize).
        # For a manual browser review instead, see "Choosing how to clean the
        # ICA" below.
        {'ica_raw': {'n_components': 0.999, 'picks': 'eeg', 'l_freq': 1}},
        {'ica_autoreject': {'eogmeasure': 'correlation', 'eogthreshold': 0.35,
                            'ecgmethod': 'ctps', 'ecgthreshold': 0.1, 'apply': False}},
        {'slice_reject': {}},
        {'ckpt_report': {'ckpt_name': 'after_ica', 'dB': False}},

        # -- 4.6  Final bad channels, interpolation, re-reference ----------
        {'bad_channels': {'picks': 'eeg', 'significance_level': 0.1}},
        {'interpolate_bads': {}},
        {'ckpt_report': {'ckpt_name': 'after_interp', 'dB': False}},
        {'set_eeg_reference': {'projection': True}},   # NATVIEW recorded ref = FCz
        {'summary': {}},
    ]
}

#%%
# Things worth internalising about this config:
#
# - **It is just an osl-ephys config.** semp adds vocabulary (new stage names), not new syntax. You can ``write_config`` / ``load_config`` it, override entries by index, and batch it exactly as in the previous tutorials.
# - **Order matters more than usual.** The gradient artefact must come off (4.2) *before* you filter and resample (4.3): its huge amplitude and sharp edges would otherwise smear across the band during filtering. And ``create_epoch`` must precede ``epoch_aas``, which reads the ``'tr_ep'`` epochs the former stashes in the ``dataset``.
# - **Why IIR, not FIR (4.3).** For the broad 0.5--125 Hz pass with a large out-of-band GA residual, a 5th-order Butterworth attenuates the stop-band better than the default FIR here. ``mid_crop`` then removes the filter's edge transients before downsampling 5000 -> 250 Hz.
# - **Why the slice rejection reuses the general ICA (4.5).** AAS assumes a single stationary template per channel; real gradients drift slightly, leaving residual power at the slice harmonics. ``slice_reject`` targets exactly those components --- on the *same* ICA that ``ica_raw`` already fitted, so there is one decomposition and one apply for EOG/ECG + slice together, not a second ICA.
# - **The BCG step is dataset-dependent.** This config leans on ``slice_reject`` + the general ICA (which picks up the strongest pulse components via the ``ECG`` channel) for pulse residuals. If you want an explicit template for a heartbeat- or pump-locked artefact, add an OBS pass after the GA removal: create the epochs, then run ``epoch_obs`` on them. ``create_He_epoch`` builds one epoch per trigger in ``dataset['he_event_key']`` and **auto-sizes the window** from the trigger spacing (``create_epoch`` with ``mode='auto'``), which is what you want when the period is not a fixed constant like the TR::
#
#       {'create_He_epoch': {}},
#       {'epoch_obs': {'epoch_key': 'he_ep', 'npc': 3, 'remove_mean': True}},

#%%
# **Choosing how to clean the ICA (4.5).** There are two ways to decide which components to remove, and which you pick changes the tail of the pipeline:
#
# - **Automatic** (shown above): ``ica_raw`` fits the decomposition and ``ica_autoreject`` labels EOG components by correlation (against ``EOGL``/``EOGU``) and ECG/BCG components by CTPS (against ``ECG``), then applies the rejection in-batch (``apply=True``). This is the right default for larger studies. If you find it misses artefacts or removes brain components, set ``apply=False`` and inspect the component topographies and time-series before deciding. With ``apply=True`` the cleaning happens inside the batch, so ``interpolate_bads`` and the average re-reference (4.6) can follow directly.
# - **Manual browser review**: replace the two ``ica_*`` stages with a single ``{'manual_ica': {...}}`` stage. ``manual_ica`` only *fits* the ICA and renders per-subject HTML review pages; it does **not** remove anything during the batch (your keep/delete decisions don't exist yet). You then review in your browser and apply the decisions in a separate step --- and because the components are removed later, ``interpolate_bads`` and the re-reference must move *after* that apply step, not stay in this config. The whole fit/review/apply workflow is the subject of the :doc:`preprocessing_manual-ica` tutorial. Manual review is most worth the effort exactly here, in EEG-fMRI, where residual GA/BCG components are hard to auto-label.

#%%
# Running the batch and reading the diagnostics
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# With the config, the pathfinder and ``initialize`` in hand, running is the familiar batch call --- just using semp's ``run_proc_batch``. Build the subject/file lists from the pathfinder, and (optionally) skip recordings already finished or errored so the batch is resumable:

if __name__ == '__main__':
    subject_list = sorted(pf.ids)                               # e.g. ['11', '21']
    file_list = [str(pf.id2path('raw', s)) for s in subject_list]

    # resumable: skip recordings that already produced a preproc fif or errored
    finished = {p.parts[-2] for p in target_pth.glob('*/*_preproc-raw.fif')}
    errored = {p.parts[-1].split('_')[0] for p in target_pth.glob('logs/*.error.log')}
    pairs = [(s, f) for s, f in zip(subject_list, file_list)
             if s not in finished and s not in errored]
    subject_list, file_list = (list(z) for z in zip(*pairs)) if pairs else ([], [])

    run_proc_batch(
        config, file_list,
        subjects=subject_list,
        outdir=str(target_pth),
        extra_funcs=[initialize],     # our only project func
        gen_report=False,             # we use semp's own ckpt/summary diagnostics
        overwrite=True,
        # dask_client=True,           # parallelise exactly as in the batch tutorial
    )

#%%
# This writes, per recording, a cleaned ``<id>_preproc-raw.fif`` under ``semp_output/<id>/`` (or, on the manual ICA path, a *fitted-but-uncleaned* preproc fif plus the saved ``<id>_ica.fif`` and the review pages).
#
# **Diagnostics.** semp's QA lives in the ``ckpt_report`` / ``init_tracer`` / ``summary`` trio rather than the standard osl-ephys HTML report (hence ``gen_report=False``). Each ``ckpt_report`` you placed in the config dumps, into ``semp_output/ckpt/<subject>/<ckpt_name>/``, the PSDs and example timecourses at that point in the pipeline --- and, where you passed ``key_to_print``, the artefact template and the removed noise for that epoch type. Placing ``ckpt_report`` either side of a stage (as we did around the AAS step) gives a literal before/after picture of what it removed --- for NATVIEW you should see the ~18 Hz slice-harmonic forest in the ``raw`` checkpoint collapse by ``after_ica``. ``init_tracer`` registers scalar metrics (band-power means, kurtosis, and the slice-harmonic ratios we added in ``initialize``) evaluated at each checkpoint, and ``summary`` plots their trajectory across the pipeline. Use these the way you used the summary report in the batch tutorial: to spot the recording whose GA didn't come off cleanly and needs a closer look.
#
# .. note::
#    Parallelise with Dask exactly as in the batch tutorial (``dask_client=True`` + a ``Client``). The semp wrappers are pure functions of the ``dataset``, so they parallelise across recordings without special handling. The one caveat is memory: NATVIEW samples at 5000 Hz and ``epoch_aas`` / ``epoch_obs`` hold the epoched data as tensors, so each worker uses a lot of RAM until the ``resample`` stage --- use fewer workers than for the clean-room MEG of the earlier tutorials.

#%%
# The manual-ICA variant of the config
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Section 4.5 used the *automatic* ICA path. For EEG-fMRI --- where residual gradient and pulse (BCG) components are hard to auto-label from EOG/ECG alone --- a **manual browser review** is often the pass most worth the effort. Taking it changes only the *tail* of the config: replace the three ICA stages of 4.5 (``ica_raw`` + ``ica_autoreject`` + ``slice_reject``) with a single ``manual_ica`` stage. ``manual_ica`` is a registered osl-ephys wrapper (just like the semp stages), so it drops into the same config by name --- no import, no ``extra_funcs``:
#
# The catch is *when* the components come off. ``manual_ica`` only **fits** the ICA and renders the per-subject browser review pages; it removes **nothing** during the batch (your keep/delete decisions don't exist yet). So the two clean-data-only stages from 4.6 --- ``interpolate_bads`` and the average re-reference --- must move *out* of this config: they have to run *after* the components are actually removed. The 4.5 + 4.6 block (everything from ``ica_raw`` onward) therefore collapses to just:

manual_ica_tail = [
    # a final bad-channel pass before fitting (as in 4.6):
    {'bad_channels': {'picks': 'eeg', 'significance_level': 0.1}},
    # FIT + render review pages only -- removes nothing in-batch. Because
    # initialize set slice_interval / tr_interval, manual_ica also renders a
    # per-component gradient-artefact score to help you spot residual-GA ICs.
    {'manual_ica': {'n_components': 0.999, 'picks': 'eeg', 'l_freq': 1}},
    {'ckpt_report': {'ckpt_name': 'after_ica_fit', 'dB': False}},
    {'summary': {}},
    # NB: NO interpolate_bads / set_eeg_reference here -- they are deferred to
    # after osl-ica-apply, where the data is finally component-cleaned.
]

# the manual-path config = stages 4.1--4.4 unchanged, then the tail above in
# place of the ICA + final blocks (4.5 + 4.6). ica_raw is the 19th stage, i.e.
# index 18, so keep everything up to (not including) it:
config_manual = {'preproc': config['preproc'][:18] + manual_ica_tail}

#%%
# Run this exactly as the automatic config above (same ``run_proc_batch`` call). Per recording it now writes a *fitted-but-uncleaned* ``<id>_preproc-raw.fif``, the fitted ``<id>_ica.fif`` and the HTML review pages --- but **no** ``<id>_after_ica-raw.fif`` yet. Producing that final cleaned file is a further three steps, done outside this batch:
#
# 1. **review** each subject in the browser (``osl-ica-review``), labelling components good / bad / unsure;
# 2. **apply** the decisions (``osl-ica-apply``), which sets the bad components in ``ica.exclude``, applies the ICA, and writes ``<id>_after_ica-raw.fif``;
# 3. run the deferred **``interpolate_bads`` + average re-reference** on that cleaned file.
#
# The exact commands, the in-browser keyboard shortcuts, the ``label.txt`` / ``bads.txt`` format, and the post-apply interpolation/re-reference snippet are all covered in the :doc:`preprocessing_manual-ica` tutorial --- read it as the direct continuation of this section. Manual review is least optional exactly here, in EEG-fMRI, where automatic component labelling is least reliable.

#%%
# Concluding remarks
# ^^^^^^^^^^^^^^^^^^
# You have taken a real, openly available simultaneous EEG-fMRI dataset from an S3 download to a cleaned, source-ready sensor recording: import the wrapper-injecting ``run_proc_batch``; declare the BIDS files with a pathfinder; *read* the TR (2.1 s), slice timing (38 unique slices, step mode ~0.055 s) --- being careful to collapse multiband duplicates and to treat it as a harmonic *locator*, not an exact period --- the volume trigger (``R128``) and the mains (60 Hz, confirmed from both provenance and the PSD) off the NATVIEW sidecars and data; declare them in ``initialize``; and assemble a config that removes the gradient artefact (``epoch_aas`` + ``slice_reject``) and the pulse artefact (general ICA / optional ``epoch_obs``) using the epoch-average-subtract recipe, all under the same machinery as the rest of osl-ephys.
#
# Two natural next steps:
#
# - :doc:`preprocessing_manual-ica` --- the fit / review / apply workflow for the manual ICA path of section 4.5: how to label the remaining components in your browser and apply the decisions to produce the final cleaned ``_after_ica-raw.fif``.
# - **Source reconstruction** --- with clean sensor data and a montage attached, EEG source recon follows the same RHINO coregistration + beamforming path as the source-recon tutorials (NATVIEW ships a T1w under each ``anat/`` folder for the head model), with the parcellation chosen to respect the (lower) rank of EEG-fMRI data after artefact and component removal.
