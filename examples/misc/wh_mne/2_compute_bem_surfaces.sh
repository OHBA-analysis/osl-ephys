#!/bin/bash

# Compute BEM surfaces using MNE

export FREESURFER_HOME=/Applications/freesurfer/7.4.1
source $FREESURFER_HOME/SetUpFreeSurfer.sh

export SUBJECTS_DIR=../data

SUBJECT=sub-03

mne watershed_bem -s $SUBJECT -d $SUBJECTS_DIR --overwrite

#plot_bem_kwargs = dict(
#    subject="sub-01",
#    subjects_dir="data",
#    brain_surfaces="white",
#    orientation="coronal",
#    slices=[50, 100, 150, 200],
#)
#mne.viz.plot_bem(**plot_bem_kwargs)
