"""Functions related to polhemus fiducials.

"""

# Authors: Mark Woolrich <mark.woolrich@ohba.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mne
from mne.io import read_info
from mne.io.constants import FIFF

import osl_ephys.source_recon.rhino.utils as rhino_utils
from osl_ephys.source_recon.rhino.coreg import get_coreg_filenames
from osl_ephys.source_recon.rhino.surfaces import get_surfaces_filenames
from osl_ephys.utils.logger import log_or_print


def get_polhemus_from_info(
    info,
    include_eeg_as_headshape=False,
    include_hpi_as_headshape=True,
):
    """Get polhemus from fif file.

    Parameters
    ----------
    info : MNE Info object
        Info object.
    include_eeg_as_headshape : bool, optional
        Should we include EEG locations as headshape points?
    include_hpi_as_headshape : bool, optional
        Should we include HPI locations as headshape points?

    Returns
    -------
    polhemus_headshape : list
        3D coordinates for each headshape point.
    polhemus_rpa : list
        3D coordinates for rpa.
    polhemus_lpa : list
        3D coordinates for lpa.
    polhemus_nasion : list
        3D coordinates for nasion.
    """

    # Lists to hold polhemus data
    polhemus_headshape = []
    polhemus_rpa = []
    polhemus_lpa = []
    polhemus_nasion = []

    # Get fiducials/headshape points
    for dig in info["dig"]:

        # Check dig is in HEAD/Polhemus space
        if dig["coord_frame"] != FIFF.FIFFV_COORD_HEAD:
            raise ValueError("{} is not in Head/Polhemus space".format(dig["ident"]))

        if dig["kind"] == FIFF.FIFFV_POINT_CARDINAL:
            if dig["ident"] == FIFF.FIFFV_POINT_LPA:
                polhemus_lpa = dig["r"]
            elif dig["ident"] == FIFF.FIFFV_POINT_RPA:
                polhemus_rpa = dig["r"]
            elif dig["ident"] == FIFF.FIFFV_POINT_NASION:
                polhemus_nasion = dig["r"]
            else:
                raise ValueError("Unknown fiducial: {}".format(dig["ident"]))
        elif dig["kind"] == FIFF.FIFFV_POINT_EXTRA:
            polhemus_headshape.append(dig["r"])
        elif dig["kind"] == FIFF.FIFFV_POINT_EEG and include_eeg_as_headshape:
            polhemus_headshape.append(dig["r"])
        elif dig["kind"] == FIFF.FIFFV_POINT_HPI and include_hpi_as_headshape:
            polhemus_headshape.append(dig["r"])

    return polhemus_headshape, polhemus_rpa, polhemus_lpa, polhemus_nasion
    

def extract_polhemus_from_info(
    fif_file,
    headshape_outfile,
    nasion_outfile,
    rpa_outfile,
    lpa_outfile,
    include_eeg_as_headshape=False,
    include_hpi_as_headshape=True,
):
    """Extract polhemus from FIF info.

    Extract polhemus fids and headshape points from MNE raw.info and write them out in the required file format for rhino (in head/polhemus space in mm).

    Should only be used with MNE-derived .fif files that have the expected digitised points held in info['dig'] of fif_file.

    Parameters
    ----------
    fif_file : string
        Full path to MNE-derived fif file.
    headshape_outfile : string
        Filename to save headshape points to.
    nasion_outfile : string
        Filename to save nasion to.
    rpa_outfile : string
        Filename to save rpa to.
    lpa_outfile : string
        Filename to save lpa to.
    include_eeg_as_headshape : bool, optional
        Should we include EEG locations as headshape points?
    include_hpi_as_headshape : bool, optional
        Should we include HPI locations as headshape points?
    """
    log_or_print("Extracting polhemus from fif info")

    # Read info from fif file
    info = read_info(fif_file)

    # Get coordinates from the fif file
    polhemus_headshape, polhemus_rpa, polhemus_lpa, polhemus_nasion = get_polhemus_from_info(
        info=info,
        include_eeg_as_headshape=include_eeg_as_headshape,
        include_hpi_as_headshape=include_hpi_as_headshape,
    )

    # Check if info is from a CTF scanner
    if info["dev_ctf_t"] is not None:
        log_or_print("Detected CTF data")

        nas = np.copy(polhemus_nasion)
        lpa = np.copy(polhemus_lpa)
        rpa = np.copy(polhemus_rpa)

        nas[0], nas[1], nas[2] = nas[1], -nas[0], nas[2]
        lpa[0], lpa[1], lpa[2] = lpa[1], -lpa[0], lpa[2]
        rpa[0], rpa[1], rpa[2] = rpa[1], -rpa[0], rpa[2]

        polhemus_nasion = nas
        polhemus_rpa = rpa
        polhemus_lpa = lpa

        # CTF data won't contain headshape points, use a dummy point to avoid errors
        polhemus_headshape = [0, 0, 0]

    # Save
    log_or_print(f"saved: {nasion_outfile}")
    np.savetxt(nasion_outfile, polhemus_nasion * 1000)
    log_or_print(f"saved: {rpa_outfile}")
    np.savetxt(rpa_outfile, polhemus_rpa * 1000)
    log_or_print(f"saved: {lpa_outfile}")
    np.savetxt(lpa_outfile, polhemus_lpa * 1000)
    log_or_print(f"saved: {headshape_outfile}")
    np.savetxt(headshape_outfile, np.array(polhemus_headshape).T * 1000)

    if info["dev_ctf_t"] is not None:
        log_or_print(f"dummy headshape points saved, overwrite {headshape_outfile} or set use_headshape=False in coregisteration")

    # Warning if 'trans' in filename we assume -trans was applied using MaxFiltering
    # This may make the coregistration appear incorrect, but this is not an issue.
    if "_trans" in fif_file:
        log_or_print("fif filename contains '_trans' which suggests -trans was passed during MaxFiltering", warning=True)
        log_or_print("This means the location of the head in the coregistration plot may not be correct", warning=True)
        log_or_print("Either use the _tsss.fif file or ignore the centroid of the head in coregistration plot", warning=True)


def save_mni_fiducials(
    fiducials_file,
    nasion_outfile,
    rpa_outfile,
    lpa_outfile,
):
    """Save MNI fiducials used to calculate sMRI fiducials.

    The file must be in MNI space with the following format:

        nas -0.5 77.5 -32.6
        lpa -74.4 -20.0 -27.2
        rpa 75.4 -21.1 -21.9

    Note, the first column (fiducial naming) is ignored but the rows must be in the above order, i.e. be (nasion, left, right).

    The order of the coordinates is the same as given in FSLeyes.

    Parameters
    ----------
    fiducials_file : str
        Full path to text file containing the sMRI fiducials.
    headshape_outfile : str
        Filename to save nasion to.
    nasion_outfile : str
        Filename to save naison to.
    rpa_outfile : str
        Filename to save rpa to.
    lpa_outfile : str
        Filename to save lpa to.
    """
    if not os.path.exists(fiducials_file):
        raise FileNotFoundError(fiducials_file)

    with open(fiducials_file, "r") as file:
        data = file.readlines()

    nas = np.array([float(x) for x in data[0].split()[1:]])
    lpa = np.array([float(x) for x in data[1].split()[1:]])
    rpa = np.array([float(x) for x in data[2].split()[1:]])

    log_or_print(f"saved: {nasion_outfile}")
    np.savetxt(nasion_outfile, nas)
    log_or_print(f"saved: {rpa_outfile}")
    np.savetxt(rpa_outfile, rpa)
    log_or_print(f"saved: {lpa_outfile}")
    np.savetxt(lpa_outfile, lpa)


def plot_polhemus_points(txt_fnames, colors=None, scales=None, markers=None, alphas=None):
    """Plot polhemus points.
    
    Parameters
    ----------
    txt_fnames : list of strings
        List of filenames containing polhemus points.
    colors : list of tuples
        List of colors for each set of points.
    scales : list of floats
        List of scales for each set of points.
    markers : list of strings
        List of markers for each set of points.
    alphas : list of floats
        List of alphas for each set of points.
    """
    plt.figure()
    ax = plt.axes(projection="3d")
    for ss in range(len(txt_fnames)):
        if alphas is None:
            alpha = 1
        else:
            alpha = alphas[ss]
        if colors is None:
            color = (0.5, 0.5, 0.5)
        else:
            color = colors[ss]
        if scales is None:
            scale = 10
        else:
            scale = scales[ss]
        if markers is None:
            marker = 1
        else:
            marker = markers[ss]

        pnts = np.loadtxt(txt_fnames[ss])
        ax.scatter(pnts[0], pnts[1], pnts[2], color=color, s=scale, alpha=alpha, marker=marker)

def delete_headshape_points(recon_dir=None, subject=None, polhemus_headshape_file=None):
    """Interactively delete headshape points.

    Shows an interactive figure of the polhemus derived headshape points in polhemus space. Points can be clicked on to delete them.

    The figure should be closed upon completion, at which point there is the option to save the deletions.

    Parameters
    ----------
    subjects_dir : string
        Directory containing the subject directories, in the directory structure used by RHINO:
    subject : string
        Subject directory name, in the directory structure used by RHINO.
    polhemus_headshape_file: string
        Full file path to get the polhemus_headshape_file from, and to save any changes to. Note that this is an npy file containing the
        (3 x num_headshapepoints) numpy array of headshape points.
        
    Notes
    -----
    We can call this in two different ways, either:

    1) Specify the subjects_dir AND the subject directory in the 
       directory structure used by RHINO:
    
            delete_headshape_points(recon_dir=recon_dir, subject=subject)
    
    or:
    
    2) Specify the full path to the .npy file containing the (3 x num_headshapepoints) numpy array of headshape points:
    
            delete_headshape_points(polhemus_headshape_file=polhemus_headshape_file)
    """

    if recon_dir is not None and subject is not None:
        coreg_filenames = get_coreg_filenames(recon_dir, subject)
        polhemus_headshape_file = coreg_filenames["polhemus_headshape_file"]
    elif polhemus_headshape_file is not None:
        polhemus_headshape_file = polhemus_headshape_file
        coreg_filenames = {'polhemus_headshape_file': polhemus_headshape_file}
    else:
        ValueError('Invalid inputs. See function\'s documentation.')
      
    polhemus_headshape_polhemus = np.loadtxt(polhemus_headshape_file)

    print("Num headshape points={}".format(polhemus_headshape_polhemus.shape[1]))
    print('Click on points to delete them.')
    print('Press "w" to write changes to the file')
    print('Press "q" to close the figure')
    sys.stdout.flush()

    def scatter_headshapes(ax, x, y, z):
        # Polhemus-derived headshape points
        color, scale, alpha, marker = "red", 8, 0.7, "o"
        ax.scatter(x, y, z, color=color, marker=marker, s=scale, alpha=alpha, picker=5)
        plt.draw()

    x = list(polhemus_headshape_polhemus[0,:])
    y = list(polhemus_headshape_polhemus[1,:])
    z = list(polhemus_headshape_polhemus[2,:])

    # Create scatter plot
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    scatter_headshapes(ax, x, y, z)

    # Define function to handle click events
    def on_click(event):
        # Get index of clicked point
        ind = event.ind

        # Remove selected points from data arrays
        print('Deleted: {}, {}, {}'.format(x[ind[0]], y[ind[0]], z[ind[0]]))
        sys.stdout.flush()
        
        x.pop(ind[0])
        y.pop(ind[0])
        z.pop(ind[0])

        # Update scatter plot
        ax.cla()
        scatter_headshapes(ax, x, y, z)

    def on_press(event):
        if event.key == 'w':
            polhemus_headshape_polhemus_new = np.array([x, y, z])
            print("Num headshape points remaining={}".format(polhemus_headshape_polhemus_new.shape[1]))
            np.savetxt(coreg_filenames["polhemus_headshape_file"], polhemus_headshape_polhemus_new)
            print('Changes saved to file {}'.format(coreg_filenames["polhemus_headshape_file"]))
        elif event.key == 'q':
            print('Closing figure')
            plt.close(fig)
                    
    # Connect click event to function
    fig.canvas.mpl_connect('pick_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_press)

    plt.show()

def remove_stray_headshape_points(outdir, subject, nose=True):
    """Remove stray headshape points.

    Removes headshape points near the nose, on the neck or far away from the head.

    Parameters
    ----------
    outdir : str
        Path to subjects directory.
    subject : str
        Subject directory name.
    noise : bool, optional
        Should we remove headshape points near the nose?
        Useful to remove these if we have defaced structurals or aren't
        extracting the nose from the structural.
    """
    filenames = get_coreg_filenames(outdir, subject)

    # Load saved headshape and nasion files
    hs = np.loadtxt(filenames["polhemus_headshape_file"])
    nas = np.loadtxt(filenames["polhemus_nasion_file"])
    lpa = np.loadtxt(filenames["polhemus_lpa_file"])
    rpa = np.loadtxt(filenames["polhemus_rpa_file"])

    if nose:
        # Remove headshape points on the nose
        remove = np.logical_and(hs[1] > max(lpa[1], rpa[1]), hs[2] < nas[2])
        hs = hs[:, ~remove]

    # Remove headshape points on the neck
    remove = hs[2] < min(lpa[2], rpa[2]) - 4
    hs = hs[:, ~remove]

    # Remove headshape points far from the head in any direction
    remove = np.logical_or(hs[0] < lpa[0] - 5, np.logical_or(hs[0] > rpa[0] + 5, hs[1] > nas[1] + 5))
    hs = hs[:, ~remove]

    # Overwrite headshape file
    log_or_print(f"overwritting: {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], hs)


def extract_polhemus_from_pos(outdir, subject, filepath):
    """Saves fiducials/headshape from a pos file.

    Parameters
    ----------
    outdir : str
        Subjects directory.
    subject : str
        Subject subdirectory/ID.
    filepath : str
        Full path to the .pos file for this subject.
        Any reference to '{subject}' (or '{0}') is replaced by the subject ID.
        E.g. 'data/{subject}/meg/{subject}_headshape.pos' with subject='sub-001'
        becomes 'data/sub-001/meg/sub-001_headshape.pos'.
    """

    # Get coreg filenames
    filenames = get_coreg_filenames(outdir, subject)

    # Load file
    if "{0}" in filepath:
        pos_file = filepath.format(subject)
    else:
        pos_file = filepath.format(subject=subject)
    log_or_print(f"Saving polhemus from {pos_file}")

    # These values are in cm in polhemus space:
    num_headshape_pnts = int(pd.read_csv(pos_file, header=None).to_numpy()[0])
    data = pd.read_csv(pos_file, header=None, skiprows=[0], delim_whitespace=True)

    # RHINO is going to work with distances in mm
    # So convert to mm from cm, note that these are in polhemus space
    data.iloc[:, 1:4] = data.iloc[:, 1:4] * 10

    # Polhemus fiducial points in polhemus space
    polhemus_nasion = data[data.iloc[:, 0].str.match("nasion")].iloc[0, 1:4].to_numpy().astype("float64").T
    polhemus_rpa = data[data.iloc[:, 0].str.match("right")].iloc[0, 1:4].to_numpy().astype("float64").T
    polhemus_lpa = data[data.iloc[:, 0].str.match("left")].iloc[0, 1:4].to_numpy().astype("float64").T

    # Polhemus headshape points in polhemus space in mm
    polhemus_headshape = data[0:num_headshape_pnts].iloc[:, 1:4].to_numpy().astype("float64").T

    # Save
    log_or_print(f"saved: {filenames['polhemus_nasion_file']}")
    np.savetxt(filenames["polhemus_nasion_file"], polhemus_nasion)
    log_or_print(f"saved: {filenames['polhemus_rpa_file']}")
    np.savetxt(filenames["polhemus_rpa_file"], polhemus_rpa)
    log_or_print(f"saved: {filenames['polhemus_lpa_file']}")
    np.savetxt(filenames["polhemus_lpa_file"], polhemus_lpa)
    log_or_print(f"saved: {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], polhemus_headshape)


def extract_polhemus_from_elc(outdir, subject, filepath, remove_headshape_near_nose=False):
    """Saves fiducials/headshape from an elc file.

    Parameters
    ----------
    outdir : str
        Subjects directory.
    subject : str
        Subject subdirectory/ID.
    filepath : str
        Full path to the .elc file for this subject.
        Any reference to '{subject}' (or '{0}') is replaced by the subject ID.
        E.g. 'data/{subject}/meg/{subject}_headshape.elc' with subject='sub-001'
        becomes 'data/sub-001/meg/sub-001_headshape.elc'.
    remove_headshape_near_nose : bool, optional
        Should we remove any headshape points near the nose?
    """

    # Get coreg filenames
    filenames = get_coreg_filenames(outdir, subject)

    # Load elc file
    if "{0}" in filepath:
        elc_file = filepath.format(subject)
    else:
        elc_file = filepath.format(subject=subject)
    log_or_print(f"Saving polhemus from {elc_file}")

    with open(elc_file, "r") as file:
        lines = file.readlines()

        # Polhemus fiducial points in polhemus space
        for i in range(len(lines)):
            if lines[i] == "Positions\n":
                nasion = np.array(lines[i + 1].split()[-3:]).astype(np.float64).T
                lpa = np.array(lines[i + 2].split()[-3:]).astype(np.float64).T
                rpa = np.array(lines[i + 3].split()[-3:]).astype(np.float64).T
                break

        # Polhemus headshape points in polhemus space
        for i in range(len(lines)):
            if lines[i] == "HeadShapePoints\n":
                headshape = np.array([l.split() for l in lines[i + 1:]]).astype(np.float64).T
                break

    if remove_headshape_near_nose:
        # Remove headshape points on the nose
        remove = np.logical_and(headshape[0] > max(lpa[0], rpa[0]), headshape[2] < nasion[2])
        headshape = headshape[:, ~remove]

    # Save
    log_or_print(f"saved: {filenames['polhemus_nasion_file']}")
    np.savetxt(filenames["polhemus_nasion_file"], nasion)
    log_or_print(f"saved: {filenames['polhemus_rpa_file']}")
    np.savetxt(filenames["polhemus_rpa_file"], rpa)
    log_or_print(f"saved: {filenames['polhemus_lpa_file']}")
    np.savetxt(filenames["polhemus_lpa_file"], lpa)
    log_or_print(f"saved: {filenames['polhemus_headshape_file']}")
    np.savetxt(filenames["polhemus_headshape_file"], headshape)


def rescale_sensor_positions(fif_file, xscale, yscale, zscale):
    """Rescale the x, y, z coordinates of sensors.

    Parameters
    ----------
    fif_file : str
        Path to fif file.
    xscale : float
        x coordinate scale factor.
    yscale : float
        y coordinate scale factor.
    zscale : float
        z coordinate scale factor.
    """

    # fif file containing the preprocessed data
    if not os.path.exists(fif_file):
        raise ValueError(f"{fif_file} is missing.")

    # Load montage
    raw = mne.io.read_raw_fif(fif_file, preload=True)
    montage = raw.get_montage()

    # Move the sensors
    for d, c in zip(montage.dig, coords):
        d["r"] = c

    # Save
    raw.set_montage(montage)
    log_or_print(f"Overwritting {fif_file} with new sensor positions")
    raw.save(fif_file, overwrite=True)
