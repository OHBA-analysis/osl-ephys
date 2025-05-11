OSL Electrophysiological Data Analysis Toolbox
==============================================

This package contains models for analysing electrophysiology data. It builds on top of the widely used MNE-Python package and contains unique analysis tools for M/EEG sensor and source space analysis. Specifically, it contains tools for:

* Multi-call MaxFilter processing.
* (pre-)processing of M/EEG data using a concise config API.
* Batch parallel processing using Dask. 
* Coregistration and volumetric source reconstruction using FSL.
* Quality assurance of M/EEG processing using HTML reports.
* Statistical significant testing (using GLM permutation testing).
* And much more!


For more information on how to use osl-ephys see the :doc:`documentation <documentation>`.

This package was developed by the Oxford Centre for Human Brain Activity (OHBA) Methods Group at the University of Oxford. Our group website is `here <https://www.psych.ox.ac.uk/research/ohba-analysis-group>`_. 

If you find this toolbox useful, please cite the following:


   **van Es, M. W., Gohil, C., Quinn, A. J., & Woolrich, M. W. (2025). osl-ephys: A Python toolbox for the analysis of electrophysiology data. Frontiers in Neuroscience, 19, 1522675.**


   **Quinn, A.J., Van Es, M.W.J., Gohil, C., & Woolrich, M.W. (2022). OHBA Software Library in Python (OSL) (0.1.1). Zenodo. https://doi.org/10.5281/zenodo.6875060**



The package heavily builds on MNE-Python, and for the source recon module, on FSL. Please also cite these packages if you use them:

   **Gramfort, A., Luessi, M., Larson, E., Engemann, D.A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., Hämäläinen, M.S. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience, 7(267):1–13. doi:10.3389/fnins.2013.00267.**


   **S.M. Smith, M. Jenkinson, M.W. Woolrich, C.F. Beckmann, T.E.J. Behrens, H. Johansen-Berg, P.R. Bannister, M. De Luca, I. Drobnjak, D.E. Flitney, R. Niazy, J. Saunders, J. Vickers, Y. Zhang, N. De Stefano, J.M. Brady, and P.M. Matthews. Advances in functional and structural MR image analysis and implementation as FSL. NeuroImage, 23(S1):208-19, 2004**

If you would like to request new features or if you're confident that you have found a bug, please create a new issue on the `GitHub issues <https://github.com/OHBA-analysis/osl-ephys/issues>`_ page.

.. |logo1| image:: https://avatars.githubusercontent.com/u/15248840?s=200&v=4
    :height: 125px
    :target: https://www.win.ox.ac.uk/research/our-locations/OHBA

.. |logo2| image:: https://www.win.ox.ac.uk/images/site-logos/oxcin_logo.png
    :height: 125px
    :target: https://www.win.ox.ac.uk/

|logo2| |logo1|

-----------------------

Contents
========

.. toctree::
   :maxdepth: 2

   Install <install>
   Documentation <documentation>
   FAQ <faq>
   API Reference <autoapi/osl_ephys/index>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
