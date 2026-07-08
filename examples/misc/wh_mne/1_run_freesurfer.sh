#!/bin/bash

export FREESURFER_HOME=/Applications/freesurfer/7.4.1
source $FREESURFER_HOME/SetUpFreeSurfer.sh

export SUBJECTS_DIR=../data

RAW_DIR=../data/raw

for NUM in 01 02 03
do
    SUBJECT=sub-$NUM
    NII=$RAW_DIR/$SUBJECT/$SUBJECT\_ses-mri_acq-mprage_T1w.nii.gz
    recon-all -i $NII -s $SUBJECT -all
done
