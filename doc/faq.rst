Frequently Asked Questions (FAQ)
================================

.. contents::
   :local:

If you have a question that's not listed here, please submit an issue to the `GitHub repository <https://github.com/OHBA-analysis/osl-ephys/issues>`_.

Installation
------------

How do I install osl-ephys?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the :doc:`installation page <install>`: osl-ephys is installed with pip into its own environment.

The config and the pipeline
---------------------------

How do I write a config?
~~~~~~~~~~~~~~~~~~~~~~~~

A config is YAML text (or a ``.yml`` file, or a Python dict) with one section per stage of the pipeline, each a list of steps with their options:

.. code-block:: yaml

   meta:
     event_codes:
       event_1: 1
       event_2: 2
   preproc:
     - filter: {l_freq: 1, h_freq: 45}
     - bad_segments: {picks: mag}
   coreg:
     - extract_surfaces: {include_nose: false}
     - extract_fiducials_and_headshape: {}
     - coregister: {use_nose: false}
   source_recon:
     - forward_model: {model: Single Layer}
     - beamform: {chantypes: mag, rank: {mag: 60}}
   parcellation:
     - parcellate: {parcellation_file: atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz}
   group:
     - sign_flip: {}

The sections are ``maxfilter``, ``preproc``, ``coreg``, ``source_recon``, ``parcellation`` and ``group``, and any of them can be left out. ``meta`` holds the event trigger codes. In ``preproc``, any method of MNE's ``Raw`` or ``Epochs`` can be named (``filter``, ``resample``, ``crop``, ...) as well as the osl-ephys steps; ``osl_ephys.pipeline.registry.registered()`` lists every step by section.

How do I run a config?
~~~~~~~~~~~~~~~~~~~~~~

On one session, in Python: ``run_chain(config, session, outdir=...)``, where the session is a dict with an ``id`` and the files the steps need (``raw_file``, ``smri_file``, ...), or just a raw file. On a set of sessions, ``run_batch(config, sessions, outdir, n_workers=4)`` runs them in parallel, with a log per session and the QC report. From the terminal, ``osl-ephys run config.yml sessions.csv outdir``, with the sessions as a CSV file with the same columns.

How do I use a custom function in the pipeline?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Define a function that takes the ``dataset`` and the step's options and returns the dataset, and pass it in ``extra_funcs``:

.. code-block:: python

   import logging
   import numpy as np

   logger = logging.getLogger("osl_ephys")

   def custom_function(dataset, userargs):
       target = userargs.pop("target", "raw")
       picks = userargs.pop("picks", "meg")
       # The logger writes to the session's log file
       logger.info(f"Taking the absolute value of the {target} data.")
       # Manipulate any key of the dataset, or add a new one
       dataset["rawabs"] = dataset[target].copy().apply_function(np.abs, picks=picks)
       # A figure in dataset["fig"] is saved and gets a tab in the report
       dataset["fig"]["rawabs_psd"] = dataset["rawabs"].compute_psd().plot(show=False)
       return dataset

   config = """
   preproc:
     - custom_function: {picks: meg}
   """
   run_chain(config, session, outdir=outdir, extra_funcs=[custom_function])

A new key of the dataset is saved under the session's name (``sub-01_rawabs.fif`` here, a pickle for other objects). The same signature works in every section: a ``coreg``, ``source_recon`` or ``parcellation`` function gets the session's files with ``Session.from_dataset(dataset)``.

Can I do group analysis with the config?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes: the ``group`` section runs after every session, on a dataset whose ``raw`` is the list of the sessions' preprocessed files and whose ``sessions`` is the manifest. ``sign_flip`` is the built-in group step; the GLM group steps (``glm_fit`` with ``run_on_group``) and custom functions work the same way as in ``preproc``.

How can I use multiple cores?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``run_batch(..., n_workers=4)`` processes four sessions at a time, each in its own process. Use fewer than half the cores of the computer, and on a shared system discuss the number with your IT support. Put the call under ``if __name__ == "__main__":`` in a script, as the worker processes import it.

Preprocessing
-------------

How do I refine the pipeline for my data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Have a look at the :doc:`tutorials_build/preprocessing_automatic` tutorial, and at the QC report: sort its table by the metric that worries you to find the sessions to look at.

What is MaxFilter and how do I use it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MaxFilter is Elekta/MEGIN licensed software for their scanners. It uses the reference sensors and Signal Space Separation (SSS) to remove external noise and compensate for head movement. In osl-ephys it is the ``maxfilter`` section of a config, run on the machine that has the licence (``osl-ephys maxfilter files.txt outdir --scanner Neo --mode multistage --tsss`` is the same thing from the terminal); see :mod:`osl_ephys.maxfilter` for the options. `MNE-Python has a maxwell filter that does not need a licence <https://mne.tools/stable/generated/mne.preprocessing.maxwell_filter.html>`_ (the ``maxwell_filter`` step). Also see the `MaxFilter user manual <https://ohba-analysis.github.io/osl-docs/downloads/maxfilter_user_guide.pdf>`_ and `these guidelines <https://lsr-wiki-01.mrc-cbu.cam.ac.uk/meg/maxpreproc>`_.

How do I select which components to remove in ICA?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Commonly the components related to heartbeats and eye movements are removed. They can be found automatically: ``ica_label`` labels them with a classifier (mne-icalabel), and ``ica_raw`` followed by ``ica_autoreject`` (or ``ica_ecg_eog_correlation``) by their correlation with the ECG and EOG channels. We recommend an automatic first pass and a manual check: ``osl-ephys ica review outdir`` opens each session's components in a GUI (click a time course to mark it, number keys label the artefact type, close the window to save), and ``osl-ephys ica apply outdir`` removes the labelled components from the data. The report's Artefacts tab shows the properties of every rejected component.

Eye and heart components are usually easy to recognise; `this advice from the FieldTrip toolbox is useful <https://www.fieldtriptoolbox.org/tutorial/ica_artifact_cleaning/#identifying-artifactual-components>`_: eye-related components are frontal, blinks and vertical saccades symmetric, horizontal saccades left-right; heart-related components in MEG are a deep bipolar source over the left and right of the helmet. Heart components are rarely seen in EEG.

How do I preprocess my OPM-MEG data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``amm`` step (adaptive multipole models) rejects interference using the sensor geometry, and ``decimate_headshape_points`` thins a dense headshape scan; see the ``examples/opm`` scripts. For OHBA's CERCA system there are scripts by Rob Seymour at https://github.com/neurofractal/OPM-oxford/tree/main/tutorials.

Source reconstruction
---------------------

How do I coregister my MRI and MEG data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Coregistration maps between the MEG (device) space, the Polhemus (head) space of the fiducials and headshape points, the native space of the structural MRI and MNI space. See the :doc:`tutorials_build/source-recon_coreg` tutorial, and the ``coreg`` section of a config (surfaces, fiducials and headshape, coregistration).

How do I use FreeSurfer for source reconstruction?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install FreeSurfer and set ``FREESURFER_HOME`` (or call ``osl_ephys.source_recon.setup_freesurfer``), then run the config with ``surface_extraction_method="freesurfer"``: ``recon_all`` and ``make_watershed_bem`` replace the surface extraction, ``coregister`` and ``forward_model`` take their FreeSurfer options, and ``minimum_norm`` with ``parcellate: {source_method: minimum_norm}`` replaces the beamformer. An example is ``examples/wakeman_henson/2_coregister_freesurfer.py`` with ``3_source_reconstruct_freesurfer.py``.

How do I refine the coregistration of a particular subject?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rerun its ``coregister`` step with a higher ``n_init`` (``run_batch(config, [session], outdir, overwrite=True)``), or remove misleading headshape points first: see the :doc:`tutorials_build/source-recon_deleting-headshape-points` tutorial.

Utilities
---------

How do I use the Study class for finding my data?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`osl_ephys.io.Study` finds the files matching a pattern with wildcards, and selects those that match a value:

.. code-block:: python

   from osl_ephys.io import Study

   study = Study('/path/to/sub{subject_id}-run{run_id}_preproc-raw.fif')
   all_files = study.get()
   subject1_files = study.get(subject_id=1)

Where are the parcellations and templates?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

They are downloaded from `osl-files <https://github.com/OHBA-analysis/osl-files>`_ on first use and cached (``osl-ephys download-data`` fetches them all, for working offline). A parcellation is named in a config by its file name, e.g. ``atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz``; :mod:`osl_ephys.files` lists them.

Other
-----

I found a bug, what do I do?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create an issue `here <https://github.com/OHBA-analysis/osl-ephys/issues>`_.

Does osl-ephys contain functionality for training generative models (e.g., HMM, DyNeMo)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No; that is `osl-dynamics <https://github.com/OHBA-analysis/osl-dynamics>`_ (`documentation <https://osl-dynamics.readthedocs.io/en/latest/>`_), which trains its models on the parcel time courses osl-ephys produces.

How can I cite the package?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

See the citation information on `GitHub <https://github.com/OHBA-analysis/osl-ephys/blob/main/CITATION.cff>`_ ("Cite this repository"). Please also cite `MNE-Python <https://github.com/mne-tools/mne-python>`_, and `FSL <https://fsl.fmrib.ox.ac.uk/fsl/docs/#/license>`_ if you used the source reconstruction.
