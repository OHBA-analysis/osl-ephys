# Manual ICA pipeline

As the automated ICA pipeline, but every session's ICA components are
labelled by hand. You need the raw `.fif` file of each session and a
structural MRI of each subject.

1. `1_maxfilter.py` runs MaxFilter (on a computer with a MaxFilter licence).
2. `2_preprocess.py` filters, downsamples, marks bad segments and channels
   and fits the ICA. The EOG and ECG detection labels components as a
   starting point but does not remove them (`apply: false`).
3. `3_review_ica.py` opens the ICA GUI on each session; the components
   labelled when the window is closed are removed from the preprocessed
   data. The same from the terminal: `osl-ephys ica review data --reject
   all`. The browser review pages (the `manual_ica` step and `osl-ephys
   ica review data --browser`) are the alternative for a remote computer.
4. `4_coregister.py` extracts the surfaces from the MRI and coregisters the
   head to them. Check the coregistration
   tab of `data/qc/report.html`; a session that looks off can be rerun
   with a higher `n_init`.
5. `5_source_reconstruct.py` computes the forward model, beamforms and
   parcellates. The parcel time courses are `data/<id>/<id>_parc-raw.fif`.
6. `6_sign_flip.py` aligns the sign of each parcel across sessions. The
   output is `data/<id>/<id>_sflip_parc-raw.fif`.
