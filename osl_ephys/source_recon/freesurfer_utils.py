"""Wrappers for Freesurfer.

"""

# Authors: Mats van Es <mats.vanes@psych.ox.ac.uk>

import os
import subprocess
import os.path as op

from mne import setup_source_space, write_source_spaces, read_source_spaces
from osl_ephys.utils.logger import log_or_print

def setup_freesurfer(directory):
    """Setup FSL.

    Parameters
    ----------
    directory : str
        Path to FSL installation.
    """
    
    os.environ["FREESURFERDIR"] = directory
    
    # Define FREESURFER_HOME
    os.environ['FREESURFER_HOME'] = directory
        
    # Source the SetUpFreeSurfer.sh script and capture the output
    setup_cmd = f"source {os.environ['FREESURFER_HOME']}/SetUpFreeSurfer.sh && env"
    proc = subprocess.Popen(setup_cmd, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    output, _ = proc.communicate()

    # Update the current environment with the new variables
    for line in output.decode().split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            os.environ[key] = value
            
    # check that it contains a license file
    if not op.exists(op.join(directory, "license.txt")):
        raise RuntimeError(f"Could not find license file in {directory}. Please visit https://surfer.nmr.mgh.harvard.edu/fswiki/License.")
    
    
    
def check_freesurfer():
    """Check FreeSurfer is installed."""
    if "FREESURFERDIR" not in os.environ:
        raise RuntimeError("Please setup FreeSurfer, e.g. with osl_ephys.source_recon.setup_freesurfer().")


def get_freesurfer_files(subjects_dir, subject):
    """Get paths to all FreeSurfer files.

    Files will be in subjects_dir/subject/.

    Parameters
    ----------
    subjects_dir : string
        Directory containing the subject directories.
    subject : string
        Subject directory name to put the coregistration files in.

    Returns
    -------
    files : dict
        A dict of files generated and used by RHINO. Contains three keys:
        - 'surf': containing surface extraction file paths.
        - 'coreg': containing coregistration file paths.
        - 'fwd_model': containing the forward model file path.
    """
    
    # Base FreeSurfer directory
    fs_dir = op.join(subjects_dir, subject)
    if " " in fs_dir:
        raise ValueError("subjects_dir cannot contain spaces.")

    # Surfaces files
    surfaces_dir = op.join(fs_dir, "surfaces")
    os.makedirs(surfaces_dir, exist_ok=True)
    surf_files = {
        "basedir": fs_dir,
        "smri_file": op.join(surfaces_dir, f"{subject.split('-')[-1]}.mgz"), # TODO: make more robust
        "talairach_xform": op.join(surfaces_dir,  "tranforms", "talairach.xfm"),
        "bem_brain_surf_file": op.join(surfaces_dir,  "bem", "brain.surf"),
        "bem_scalp_surf_fif": op.join(surfaces_dir, "bem", f"{subject}-head.fif"),
        "bem_inner_skull_surf_file": op.join(surfaces_dir,  "bem", "inner_skull.surf"),
        "bem_outer_skull_surf_file": op.join(surfaces_dir,  "bem", "outer_skull.surf"),
        "bem_outer_skin_surf_file": op.join(surfaces_dir,  "bem", "outer_skin.surf"),
        "bem_ws_brain_surf_file": op.join(surfaces_dir, "bem", "watershed", f"{subject}_brain_surface"),
        "bem_ws_inner_skull_surf_file": op.join(surfaces_dir, "bem", "watershed", f"{subject}_inner_skull_surface"),
        "talairach_xform": op.join(surfaces_dir,  "tranforms", "talairach.xfm"),
        "std_brain_dir": op.join(os.environ["FREESURFER_HOME"], "subjects", "fsaverage"),
        "std_brain_mri": op.join(os.environ["FREESURFER_HOME"], "subjects", "fsaverage", "mri", "T1.mgz"),
        "completed": op.join(surfaces_dir, "completed.txt"),
    }
    #     "mni2mri_flirt_xform_file": op.join(surfaces_dir, "tranforms", "talairach.xfm"),
    #     "mni_mri_t_file": op.join(surfaces_dir, "mni_mri-trans.fif"),
    #     "bet_outskin_mesh_vtk_file": op.join(surfaces_dir, "outskin_mesh.vtk"),  # BET output
    #     "bet_inskull_mesh_vtk_file": op.join(surfaces_dir, "inskull_mesh.vtk"),  # BET output
    #     "bet_outskull_mesh_vtk_file": op.join(surfaces_dir, "outskull_mesh.vtk"),  # BET output
    #     "bet_outskin_mesh_file": op.join(surfaces_dir, "outskin_mesh.nii.gz"),
    #     "bet_outskin_plus_nose_mesh_file": op.join(surfaces_dir, "outskin_plus_nose_mesh.nii.gz"),
    #     "bet_inskull_mesh_file": op.join(surfaces_dir, "inskull_mesh.nii.gz"),
    #     "bet_outskull_mesh_file": op.join(surfaces_dir, "outskull_mesh.nii.gz"),
    #     "std_brain": op.join(os.environ["FSLDIR"], "data", "standard", "MNI152_T1_1mm_brain.nii.gz"),
    #     "std_brain_bigfov": op.join(os.environ["FSLDIR"], "data", "standard", "MNI152_T1_1mm_BigFoV_facemask.nii.gz"),
    #     "completed": op.join(surfaces_dir, "completed.txt"),
    # }

    # Coregistration files
    coreg_dir = op.join(fs_dir, "mne_src")
    os.makedirs(coreg_dir, exist_ok=True)
    coreg_files = {
        "basedir": coreg_dir,
        "info_fif_file": op.join(coreg_dir, "info-raw.fif"),
        "coreg_trans": op.join(coreg_dir, "coreg-trans.fif"),
        "coreg_html": op.join(coreg_dir, "coreg.html"),
        "source_space": op.join(coreg_dir, "space-src.fif"),
    }
        
        
        # "info_fif_file": op.join(coreg_dir, "info-raw.fif"),
        # "smri_file": op.join(coreg_dir, "scaled_smri.nii.gz"),
        # "head_scaledmri_t_file": op.join(coreg_dir, "head_scaledmri-trans.fif"),
        # "head_mri_t_file": op.join(coreg_dir, "head_mri-trans.fif"),
        # "ctf_head_mri_t_file": op.join(coreg_dir, "ctf_head_mri-trans.fif"),
        # "mrivoxel_scaledmri_t_file": op.join(coreg_dir, "mrivoxel_scaledmri_t_file-trans.fif"),
        # "mni_nasion_mni_file": op.join(coreg_dir, "mni_nasion.txt"),
        # "mni_rpa_mni_file": op.join(coreg_dir, "mni_rpa.txt"),
        # "mni_lpa_mni_file": op.join(coreg_dir, "mni_lpa.txt"),
        # "smri_nasion_file": op.join(coreg_dir, "smri_nasion.txt"),
        # "smri_rpa_file": op.join(coreg_dir, "smri_rpa.txt"),
        # "smri_lpa_file": op.join(coreg_dir, "smri_lpa.txt"),
        # "polhemus_nasion_file": op.join(coreg_dir, "polhemus_nasion.txt"),
        # "polhemus_rpa_file": op.join(coreg_dir, "polhemus_rpa.txt"),
        # "polhemus_lpa_file": op.join(coreg_dir, "polhemus_lpa.txt"),
        # "polhemus_headshape_file": op.join(coreg_dir, "polhemus_headshape.txt"),
        # # BET mesh output in native space
        # "bet_outskin_mesh_vtk_file": op.join(coreg_dir, "scaled_outskin_mesh.vtk"),
        # "bet_inskull_mesh_vtk_file": op.join(coreg_dir, "scaled_inskull_mesh.vtk"),
        # "bet_outskull_mesh_vtk_file": op.join(coreg_dir, "scaled_outskull_mesh.vtk"),
        # # Freesurfer mesh in native space
        # # - these are the ones shown in coreg_display() if doing surf plot
        # # - these are also used by MNE forward modelling
        # "bet_outskin_surf_file": op.join(coreg_dir, "scaled_outskin_surf.surf"),
        # "bet_inskull_surf_file": op.join(coreg_dir, "scaled_inskull_surf.surf"),
        # "bet_outskull_surf_file": op.join(coreg_dir, "scaled_outskull_surf.surf"),
        # "bet_outskin_plus_nose_surf_file": op.join(coreg_dir, "scaled_outskin_plus_nose_surf.surf"),
        # # BET output surface mask as nii in native space
        # "bet_outskin_mesh_file": op.join(coreg_dir, "scaled_outskin_mesh.nii.gz"),
        # "bet_outskin_plus_nose_mesh_file": op.join(coreg_dir, "scaled_outskin_plus_nose_mesh.nii.gz"),
        # "bet_inskull_mesh_file": op.join(coreg_dir, "scaled_inskull_mesh.nii.gz"),
        # "bet_outskull_mesh_file": op.join(coreg_dir, "scaled_outskull_mesh.nii.gz"),
        # "std_brain": op.join(os.environ["FSLDIR"], "data", "standard", "MNI152_T1_1mm_brain.nii.gz"),
    

    # All Freesurfer files files
    files = {"surf": surf_files, "coreg": coreg_files, "fwd_model": op.join(fs_dir, "model-fwd.fif")}

    return files



def get_coreg_filenames(subjects_dir, subject):
    """Files used in coregistration by FreeSurfer.

    Files will be in subjects_dir/subject/.

    Parameters
    ----------
    subjects_dir : string
        Directory containing the subject directories.
    subject : string
        Subject directory name to put the coregistration files in.

    Returns
    -------
    filenames : dict
        A dict of files generated and used by FreeSurfer.
    """
    fs_files = get_freesurfer_files(subjects_dir, subject)
    return fs_files["coreg"]


def make_fsaverage_src(subjects_dir, ico=5):
    subject = 'fsaverage'
    src_fname = get_coreg_filenames(subjects_dir, subject)['source_space']
    
    if not op.exists(src_fname):
        src = setup_source_space(
        subjects_dir=subjects_dir,
        subject=subject,
        spacing="oct6",
        add_dist="patch",
        ico=ico,
        )
        write_source_spaces(src_fname, src)
    else:
        src = read_source_spaces(src_fname) 
        
    return src