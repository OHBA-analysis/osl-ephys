"""Reviewing ICA components by hand, and applying the decisions.

Two routes label the components of a session's ICA:

- the GUI (:func:`plot_ica`): :func:`review_gui` opens the session's
  components, the labels are saved in the ICA file;
- the browser pages of :mod:`~osl_ephys.preproc.ica.manual`, written by the
  ``manual_ica`` step and served by :func:`serve_review`, which save
  ``label.txt`` and ``bads.txt`` under ``outdir/ica/<id>/``.

:func:`apply_review` removes the labelled components from the session's
preprocessed data on either route. On the command line: ``osl-ephys ica
review`` and ``osl-ephys ica apply``.
"""

from osl_ephys.preproc.ica.gui import plot_ica
from osl_ephys.preproc.ica.review import review_gui, serve_review
from osl_ephys.preproc.ica.apply import apply_review
