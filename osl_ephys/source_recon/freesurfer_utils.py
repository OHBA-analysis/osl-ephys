"""Wrappers for Freesurfer.

"""

# Authors: Mats van Es <mats.vanes@psych.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import subprocess
import os.path as op

from mne import setup_source_space, write_source_spaces, read_source_spaces, bem
from osl_ephys.utils.logger import log_or_print

def setup_freesurfer(directory, subjects_dir=None):
    """Setup FreeSurfer.

    Parameters
    ----------
    directory : str
        Path to FreeSurfer installation.
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
    
    # Set subjects_dir
    if subjects_dir is not None:
        os.environ["SUBJECTS_DIR"] = subjects_dir
    
    
    
def check_freesurfer():
    """Check FreeSurfer is installed."""
    if "FREESURFERDIR" not in os.environ:
        raise RuntimeError("Please setup FreeSurfer, e.g. with osl_ephys.source_recon.setup_freesurfer().")


def get_freesurfer_filenames(subjects_dir, subject):
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
    
    # Coregistration files
    coreg_dir = op.join(fs_dir, "coreg")
    os.makedirs(coreg_dir, exist_ok=True)
    coreg_files = {
        "basedir": coreg_dir,
        "info_fif_file": op.join(coreg_dir, "info-raw.fif"),
        "source_space": op.join(coreg_dir, "space-src.fif"),
        "coreg_trans": op.join(coreg_dir, "coreg-trans.fif"),
        "coreg_html": op.join(coreg_dir, "coreg.html"),
    }

    # Forward model filename
    fwd = op.join(fs_dir, "model-fwd.fif")

    # All Freesurfer files
    files = {"surf": surf_files, "coreg": coreg_files, "fwd_model": fwd}

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
    fs_files = get_freesurfer_filenames(subjects_dir, subject)
    return fs_files["coreg"]


def recon_all(smri_file, subjects_dir, subject):

    os.environ["SUBJECTS_DIR"] = subjects_dir
    
    move_flag = False
    if os.path.exists(os.path.join(subjects_dir, subject)):
        log_or_print(f'Temporarily saving data to {op.join(subjects_dir, subject + "_freesurfer_temp")} because subject {subject} already exists')
        cmd =  ['recon-all', '-i', smri_file, '-s', subject + '_freesurfer_temp', '-all'] 
        move_flag = True
    else:
        cmd = ['recon-all', '-i', smri_file, '-s', subject, '-all'] 
    
    try:
        subprocess.run(cmd, check=True, env=os.environ)
        log_or_print(f"recon-all completed successfully for subject {subject}")
    except subprocess.CalledProcessError as e:
        log_or_print(f"Error running recon-all for subject {subject}: {e}")

    if move_flag:
        log_or_print(f'Moving data from {op.join(subjects_dir, subject + "_freesurfer_temp")} to {op.join(subjects_dir, subject)}')
        os.rename(op.join(subjects_dir, subject + "_freesurfer_temp"), op.join(subjects_dir, subject))


def make_watershed_bem(outdir, subject, **kwargs):
    """Wrapper for :py:func:`mne.bem.make_watershed_bem <mne.bem.make_watershed_bem>` making a watershed BEM with FreeSurfer."""   
    
    check_freesurfer()
    
    bem.make_watershed_bem(
        subject=subject,
        subjects_dir=outdir,
        **kwargs
    )


def make_fsaverage_src(subjects_dir, spacing='oct6'):

    subject = 'fsaverage'
    src_fname = get_coreg_filenames(subjects_dir, subject)['source_space']
    
    if not op.exists(src_fname):
        src = setup_source_space(
            subjects_dir=subjects_dir,
            subject=subject,
            spacing=spacing,
            add_dist="patch",
        )
        write_source_spaces(src_fname, src)
