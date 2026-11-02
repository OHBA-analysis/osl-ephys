Documentation
=============

The :doc:`API reference <autoapi/osl_ephys/index>` documents every module,
class and function. New users may find the :doc:`FAQ <faq>` useful.

The pipeline
------------

A config names the steps of each stage (``maxfilter``, ``preproc``,
``coreg``, ``source_recon``, ``parcellation``, ``group``) and
:func:`~osl_ephys.pipeline.run_batch` runs it on a set of sessions in
parallel, writing a log per session and a QC report:

.. code-block:: python

   from osl_ephys.pipeline import run_batch

   config = """
   preproc:
     - filter: {l_freq: 1, h_freq: 45}
     - resample: {sfreq: 250}
     - bad_segments: {picks: mag}
     - bad_channels: {picks: mag}
     - ica_label: {picks: mag}
   coreg:
     - extract_surfaces: {include_nose: false}
     - extract_fiducials_and_headshape: {}
     - coregister: {use_nose: false}
   source_recon:
     - forward_model: {model: Single Layer}
     - beamform: {chantypes: mag, rank: {mag: 60}}
   parcellation:
     - parcellate: {parcellation_file: atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz}
   """

   sessions = [
       {"id": "sub-01", "subject": "sub-01", "raw_file": "raw/sub-01.fif", "smri_file": "smri/sub-01.nii.gz"},
       {"id": "sub-02", "subject": "sub-02", "raw_file": "raw/sub-02.fif", "smri_file": "smri/sub-02.nii.gz"},
   ]

   run_batch(config, sessions, "derivatives", n_workers=2)

The same run from the command line, with the sessions in a CSV file::

   osl-ephys run config.yml sessions.csv derivatives --n-workers 2

Tutorials
---------

**Preprocessing**:

- :doc:`tutorials_build/preprocessing_manual`
- :doc:`tutorials_build/preprocessing_automatic`
- :doc:`tutorials_build/preprocessing_batch`

**Preprocessing simultaneous EEG-fMRI** (the ``semp`` subpackage and the
manual ICA review):

- :doc:`tutorials_build/preprocessing_eeg-fmri`
- :doc:`tutorials_build/preprocessing_manual-ica`

**Source reconstruction**:

- :doc:`tutorials_build/source-recon_coreg`
- :doc:`tutorials_build/source-recon_deleting-headshape-points`
- :doc:`tutorials_build/source-recon_subject`
- :doc:`tutorials_build/source-recon_batch`

**Statistics (general linear modelling)**:

- :doc:`tutorials_build/glm_simulation`
- :doc:`tutorials_build/glm_sensor`

Example scripts for each scanner, the pipelines used at OHBA and the
analysis of the toolbox paper are in the `examples directory
<https://github.com/OHBA-analysis/osl-ephys/tree/main/examples>`_ of the
repository.

Workshops
---------

- `OHBA Software Library (OSL) workshop <https://osf.io/zxb6c/>`_.
