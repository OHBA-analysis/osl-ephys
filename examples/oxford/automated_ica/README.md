# Automated ICA pipeline

The recommended first pass for recordings from the OHBA TRIUX (Neo): the
eye and heart artefacts are removed by ICA with the components labelled
automatically. You need the raw `.fif` file of each session, with EOG and
ECG channels, and a structural MRI of each subject.

1. `1_maxfilter.py` runs MaxFilter (on a computer with a MaxFilter licence).
2. `2_preprocess.py` filters, downsamples, marks bad segments and channels
   and removes the eye and heart artefacts with ICA. Check the
   preprocessing tabs of `data/qc/report.html`; the automatic labelling is
   not always right, and the components of a session can be relabelled by
   hand with `osl-ephys ica review data <id> --reject all`.
3. `3_coregister.py` extracts the surfaces from the MRI and coregisters the
   head to them. Check the coregistration
   tab of the report; a session that looks off can be rerun with a higher
   `n_init`.
4. `4_source_reconstruct.py` computes the forward model, beamforms and
   parcellates. The parcel time courses are `data/<id>/<id>_parc-raw.fif`.
5. `5_sign_flip.py` aligns the sign of each parcel across sessions, which
   matters when a group model (such as an HMM) is trained on the data. The
   output is `data/<id>/<id>_sflip_parc-raw.fif`.
