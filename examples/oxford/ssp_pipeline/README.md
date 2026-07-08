SSP Pipeline
------------

The scripts in this directory contain the recommended settings for preprocessing continuous data collected using the new scanner (Neo) at OHBA, Oxford. It doesn't matter if the data was recorded during a task or at rest.

To run these scripts you need:

1. The raw `.fif` file containing the recording for each session.

2. A structural MRI for each subject.

The steps are:

1. **MaxFilter** (`1_maxfilter.py`). This script MaxFilters the raw recordings.

2. **Preprocess** (`2_preprocess.py`). This script filters, downsamples, and uses automated algorithms to detect bad segments/channels and uses ICA to remove eye blinks. Note, the automated ICA artefact removal might not always work very well, so it's often worthwhile to check the preprocessing.

3. **SSP denoising** (`3_ssp_denoising.py`). This script can be used to do SSP artefact rejection.

4. **Coregistration** (`4_coregister.py`). This script aligns the sMRI and MEG space (using the polhemus head space). Here, we advise you check the head has been placed in a plausible location in the MEG scanner. You can do this by looking at the coregistration panel in the report (`coreg/report/subjects_report.html` and `summary_report.html`). You may need to re-run this script with different settings (e.g. by increasing `n_init`) to fix poorly aligned subjects.

5. **Source reconstruct** (`5_source_reconstruct.py`). This script calculates the forward model, beamforms and parcellates the data. The parcellated data files can be found in `src/{subject}/parc/parc-raw.fif`.

6. **Dipole sign flipping** (`6_sign_flip.py`). This script tries to minimise the effect of the sign of each parcel time course being misaligned between subjects when fitting group-level models, such as a Hidden Markov Model. This step maybe skipped if you're not interested in fitting group models.

The final data can be found in `src/{subject}/sflip_parc-raw.fif`.
