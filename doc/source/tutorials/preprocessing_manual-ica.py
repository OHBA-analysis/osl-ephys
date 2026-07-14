'''
Manual ICA review in the browser
================================

In the :doc:`preprocessing_automatic` tutorial you saw osl-ephys flag and remove artefactual ICA components *automatically*, using EOG/ECG correlations. That is the right default for large, clean MEG datasets. But there are situations where you want a human in the loop:

- **Noisy or unusual data** --- e.g. simultaneous EEG-fMRI, where residual gradient and pulse artefacts produce components that no simple EOG/ECG correlation will catch.
- **Small studies** where you can afford to eyeball every subject and want full control over what is removed.
- **Method development / QA**, where you need to *see* exactly which components were dropped and why.

``osl_ephys.preprocessing.manual_ica`` supports this with a **browser-based** review tool. It fits the ICA during your normal preprocessing batch and renders a self-contained web page per subject; you then click through the components in your browser, label each one ``good`` / ``bad`` / ``unsure``, and a small command-line tool applies your decisions. No notebook, no blocking matplotlib window --- the review is just static files plus a tiny local server, which means it works fine over SSH and can be done long after (and on a different machine from) the batch run.

This tutorial looks as follows:

1. **The fit / review / apply split**
2. **Adding** ``manual_ica`` **to a config**
3. **Reviewing components in the browser**
4. **Applying the decisions:** ``osl-ica-apply``
5. **Programmatic access to the labels**
6. **Concluding remarks**

.. note::
   As with the EEG-fMRI tutorial, the snippets here are **illustrative** --- the interesting part is an interactive browser step and a per-subject HTML page, neither of which renders in a static gallery. Point the paths at your own preprocessed data to try it for real.

'''

#%%
# The fit / review / apply split
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Manual review is inherently a three-step process, because a human has to act *in the middle*. ``manual_ica`` makes that split explicit:
#
# 1. **Fit (inside the batch).** The ``manual_ica`` preprocessing stage fits an ICA on your data, scores each component (EOG/ECG correlations, variance, and --- for EEG-fMRI --- residual gradient-artefact power), renders a diagnostic SVG per component, and writes two HTML review pages. Crucially it **does not remove anything**: at batch time your keep/delete decisions don't exist yet. The batch therefore outputs a preprocessed-but-uncleaned ``<subject>_preproc-raw.fif`` alongside the fitted ``<subject>_ica.fif`` and the review pages.
# 2. **Review (in your browser).** You serve the review folder, click through the components, and label them. Your labels are saved to plain-text ``label.txt`` (and ``bads.txt`` for any bad time-segments you mark) per subject.
# 3. **Apply (a separate command).** ``osl-ica-apply`` reads ``label.txt`` + ``bads.txt``, sets the bad components in ``ica.exclude``, applies the ICA to the preproc fif, and writes the final cleaned ``<subject>_after_ica-raw.fif``.
#
# This is why, in the EEG-fMRI config from the previous tutorial, the batch *ended* at ``manual_ica`` and the average re-reference / interpolation were deferred --- they belong *after* the components have actually been removed, i.e. in (or after) the apply step.

#%%
# Adding ``manual_ica`` to a config
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``manual_ica`` is a registered osl-ephys wrapper, so it works in a **plain** ``osl_ephys.preprocessing.run_proc_batch`` config, by name, with no special import or ``extra_funcs`` entry (the semp EEG-fMRI wrappers are registered the same way):

config = {
    'preproc': [
        # ... your filtering / bad-segment / bad-channel stages ...
        {'manual_ica': {'n_components': 0.999,
                        'picks': 'eeg',
                        'l_freq': 1.0}},
    ]
}

#%%
# The one requirement is that ``dataset['subject']`` is set by an upstream stage (it names the per-subject review folder). In a plain osl-ephys pipeline ``subject`` is already threaded through ``run_proc_batch(..., subjects=[...])``; in a semp pipeline the project ``initialize`` extra_func sets it (see the previous tutorial). Note that ``subject`` is deliberately **rejected** as a userarg, so that a shared batch ``userargs`` dict can't make every subject write to the same folder.
#
# The full userarg schema lives in ``osl_ephys.preprocessing.manual_ica.config.DEFAULT_USERARGS`` and is enforced strictly (an unknown key raises immediately). The ones you will actually reach for:
#
# - **ICA fit**: ``n_components`` (float = variance fraction, int = count; default ``0.999``), ``method`` (``'fastica'``), ``l_freq`` (high-pass applied before fitting; ``1.0`` Hz), ``picks`` (``'eeg'``), ``seed``.
# - **Auto-flag thresholds** (these *pre-suggest* labels; you still decide): ``ecg_threshold``, ``eog_threshold``, and ``ga_threshold`` for the EEG-fMRI gradient-artefact score. ``ecg_ch`` / ``eog_ch`` let you name the reference channels explicitly instead of auto-picking from ``raw.info``.
# - **Plotting knobs**: ``spec_freq_max`` (top of the spectrum panel, 45 Hz), ``zoom_window`` (length in seconds of each zoomed timecourse window, 20 s), ``psd_resolution``, ``spec_type`` (``'linear'`` / ``'db'``).
# - **Output**: ``outdir`` --- the *root* under which per-subject review folders are created. If omitted it falls back to ``dataset['target_pth']/ica`` then the cwd. It is safe to set in shared batch userargs precisely because it is the root, not a per-subject path.
#
# .. note::
#    The gradient-artefact (``GA``) score only appears if ``dataset['slice_interval']`` and ``dataset['tr_interval']`` are set (i.e. EEG-fMRI data). For ordinary EEG/MEG those keys are absent and the GA score is silently skipped --- ``ga_threshold`` then has no effect.

#%%
# Reviewing components in the browser
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# After the batch, each subject has a folder under the review root containing the per-component SVGs and two pages: ``single_ic.html`` (one component at a time, with topography, spectrum, variance and full + zoomed timecourse) and ``between_ic.html`` (several components stacked, for comparing them against each other). Because browsers won't load sibling files straight off disk, serve the folder with the bundled server (a thin wrapper that also handles saving your labels back):
#
# .. code-block:: bash
#
#     # run from the review root (the folder that contains the per-subject dirs)
#     osl-ica-review 8000
#
#     # then open, per subject:
#     #   http://localhost:8000/<subject>/single_ic.html
#
# Working over SSH, forward the port (``ssh -L 8000:localhost:8000 ...``) and open the URL on your laptop --- the heavy rendering already happened during the batch, so the review itself is light.
#
# In the page you step through components and label them with the keyboard. The labels are the three you'd expect:
#
# - **good** --- keep (brain or otherwise harmless).
# - **bad** --- remove this component when applying.
# - **unsure** --- kept (not removed), but counted separately so you can come back to it.
#
# You can also mark **bad time-segments** (a modal that writes start/stop times), which are applied as ``BAD_manual`` annotations at the apply step. The on-page key legend lists the navigation/label bindings; any auto-suggested labels from the thresholds above are shown as a starting point that you confirm or override. Your decisions are written, per subject, to ``label.txt`` (one ``IC<nnn>: good|bad|unsure|unlabeled`` line per component) and ``bads.txt`` (the bad segments). Any component you never touch stays ``unlabeled`` --- which, as we'll see, blocks the apply step until you finish.

#%%
# Applying the decisions: ``osl-ica-apply``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Once a subject is fully labelled, apply the decisions with the bundled command-line tool:
#
# .. code-block:: bash
#
#     osl-ica-apply <ica_root> <raw_root> [subject ...] [--overwrite] [--purge-svgs]
#
# where ``<ica_root>`` is the review root (holding ``<subject>/label.txt``) and ``<raw_root>`` holds ``<subject>/<subject>_preproc-raw.fif`` and ``<subject>/<subject>_ica.fif``. With no subjects listed it processes every subject under ``<ica_root>`` that has a ``label.txt``. For each one it:
#
# 1. parses ``label.txt`` and ``bads.txt``;
# 2. **refuses to proceed** if the review isn't finished (any ``unlabeled`` component) or if ``label.txt`` has malformed lines or references a component index beyond the fit --- it prints a ``skip --- ...`` reason rather than silently doing the wrong thing;
# 3. loads the preproc fif, appends the bad segments as ``BAD_manual`` annotations, sets the bad components in ``ica.exclude``, applies the ICA, and saves ``<subject>_after_ica-raw.fif``.
#
# The ``--purge-svgs`` flag deletes the (large) per-component SVGs and the HTML pages after a successful apply, keeping the small ``label.txt`` / ``bads.txt`` so the decisions remain reproducible --- this frees a lot of disk per subject on big studies. Use ``--overwrite`` to regenerate an existing ``_after_ica-raw.fif``.
#
# After this step the ``_after_ica-raw.fif`` is your final, component-cleaned sensor data. Stages that should only run on cleaned data --- interpolating bad channels, applying an average reference --- belong here, after the apply, not in the fit batch:

import mne

cleaned = mne.io.read_raw_fif(
    '/path/<subject>/<subject>_after_ica-raw.fif', preload=True)
cleaned.interpolate_bads()
cleaned.set_eeg_reference('average', projection=True)
cleaned.apply_proj()
cleaned.save('/path/<subject>/<subject>_after_ica-raw.fif', overwrite=True)

#%%
# Programmatic access to the labels
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# If you want to drive the apply step from your own script --- to add bespoke logging, custom subject filtering, or to fold it into a larger pipeline --- use the package's public API rather than re-parsing the text files yourself (a hand-rolled regex is exactly the kind of thing that silently drifts out of sync):

from osl_ephys.preprocessing.manual_ica import (
    parse_label_txt, parse_bads_txt, apply_manual_ica)

# Inspect one subject's decisions without applying anything:
bad_ics, n_unsure, n_unlabeled, warnings = parse_label_txt(
    '/path/ica_root/<subject>/label.txt')
bad_segments = parse_bads_txt('/path/ica_root/<subject>/bads.txt')

# Apply one subject programmatically (same logic as the CLI, returns a
# human-readable status string: 'ok ...' / 'skip ...'):
status = apply_manual_ica(
    '/path/ica_root', '/path/raw_root', '<subject>',
    overwrite=True, purge_svgs=False)
print(status)

#%%
# ``parse_label_txt`` returning a non-empty ``warnings`` or a non-zero ``n_unlabeled`` is your signal that a subject's review isn't clean/finished --- handy for a quick "who still needs reviewing?" sweep across a study before you batch-apply.

#%%
# Concluding remarks
# ^^^^^^^^^^^^^^^^^^
# You have seen the manual-ICA workflow end to end: ``manual_ica`` *fits* the decomposition and renders browser review pages during your normal preprocessing batch; you *review* and label the components in your browser via ``osl-ica-review``; and ``osl-ica-apply`` (or ``apply_manual_ica`` in Python) *applies* your decisions to produce the final cleaned ``_after_ica-raw.fif``. The deliberate fit/review/apply split is what lets the human step happen whenever --- and wherever --- is convenient, without holding a Python process open.
#
# This pairs naturally with the :doc:`preprocessing_eeg-fmri` tutorial, where automatic component labelling is least reliable and a manual pass is most worth the effort; but nothing about the tool is EEG-fMRI specific --- it is a general-purpose manual ICA reviewer for any osl-ephys pipeline.
