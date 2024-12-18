"""Run group analysis on parcellated data on the Wakeman-Henson dataset.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>

import os
import os.path as op
from osl_ephys import source_recon
import numpy as np

subjects_dir = "/ohba/pi/mwoolrich/datasets/WakemanHenson/ds117"

nsubjects = 19
nsessions = 6
subjects_to_do = np.arange(0, nsubjects)
sessions_to_do = np.arange(0, nsessions)
subj_sess_2exclude = np.zeros([nsubjects, nsessions]).astype(bool)

#subj_sess_2exclude = np.ones(subj_sess_2exclude.shape).astype(bool)
#subj_sess_2exclude[0:1,0:1]=False

# ----------------
# Setup file names

smri_files = []
preproc_fif_files = []
sflip_parc_files = []

subjects = []

recon_dir = op.join(subjects_dir, "recon_fs")

for sub in subjects_to_do:
    for ses in sessions_to_do:
        if not subj_sess_2exclude[sub, ses]:

            sub_name = "sub" + ("{}".format(subjects_to_do[sub] + 1)).zfill(3)
            ses_name = "run_" + ("{}".format(sessions_to_do[ses] + 1)).zfill(2)
            subject = sub_name + "_" + ses_name

            # input files
            smri_file = op.join(subjects_dir, sub_name, "anatomy", "highres001.nii.gz")

            preproc_fif_file = op.join(
                subjects_dir, subject + "_meg", subject + "_meg_preproc_raw.fif"
            )

            # output files
            sflip_parc_file = op.join(recon_dir, subject, "sflip_parc-raw.fif")

            if op.exists(preproc_fif_file) and op.exists(smri_file):
                subjects.append(subject)
                smri_files.append(smri_file)
                preproc_fif_files.append(preproc_fif_file)
                sflip_parc_files.append(sflip_parc_file)

# -------------------------------------
# Coreg and Source recon and Parcellate

config = """
    source_recon:
    - recon_all: 
    - make_watershed_bem:
    - coregister:
        mode: mne
        nasion_weight; 2.0
    - forward_model:
        model: Single Layer
        mode: mne
        ico: 5
    - minimum_norm:
        method: eLORETA
        rank: 60
        loose: 0.2
        lambda2: 0.1
        morph: fsaverage
    - parcellate:
      
"""

# parcellation_file: fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz
# parcellation_file: Schaefer2018_100Parcels_7Networks_order_FSLMNI152_2mm_4d_ds8.nii.gz
# parcellation_file: HarvOxf-sub-Schaefer100-combined-2mm_4d_ds8.nii.gz

source_recon.run_src_batch(
    config,
    src_dir=recon_dir,
    subjects=subjects,
    preproc_files=preproc_fif_files,
    smri_files=smri_files,
    
)
