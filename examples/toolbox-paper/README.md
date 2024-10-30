# OSL Toolbox Paper

The scripts for the toolbox paper can be found here in the order they appear in the manuscript. You can download the data from [OpenfMRI](https://openfmri.org/s3-browser/?prefix=ds000117/ds000117_R0.1.1/compressed/). Extract the `tar.gz` files in a folder called `ds117`. Note: you may have to change the base directory in the scripts to match the directory where you store the data. You also need to [install FSL](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index), and make sure that `source_recon.setup_fsl()` in `2_source-reconstruct.py` and `3_sign-flip.py` is pointing to the correct directory.

If you wish to fully reproduce the analysis pipeline install the environment specified in `osl-toolbox-paper.yml`, and set `random_seed` in `run_proc_batch` and `run_src_batch` according to the seed found in the logfiles on [OSF](https://osf.io/2rnyg/).

## Manual preprocessing

Note that in the toolbox paper, automatically labeled ICA components were manually refined for the following sessions:
- sub008-ses03
- sub019-ses01
- sub019-ses02
- sub019-ses03
- sub019-ses04
- sub019-ses05
- sub019-ses06
- sub010-ses05

This was done by running the following command line function iteratively, replacing "session". 
`osl_ica_label None processed "session"`
After all specified sessions were refined, all automatically/manually labeled components were removed from the preprocessed MEG data using the command line call
`osl_ica_apply processed`