"""Utility function for handling OPM data."""

import os.path
from shutil import copyfile
import nibabel as nib

import numpy as np

import mne
from mne.io.constants import FIFF
from mne.transforms import Transform, apply_trans

try:
    from mne._fiff.tag import _coil_trans_to_loc
except ImportError:
    # Depreciated in mne 1.6
    from mne.io import _coil_trans_to_loc

import pandas as pd
import scipy


def convert_notts(notts_opm_mat_file, smri_file, tsv_file, fif_file, smri_fixed_file):
    """Convert Nottingham OPM data from matlab file to fif file.

    Parameters
    ----------
    notts_opm_mat_file : str
        The matlab file containing the OPM data.
    smri_file : str
        The structural MRI file.
    tsv_file : str
        The tsv file containing the sensor locations and orientations.
    fif_file : str
        The output fif file.
    smri_fixed_file : str
        The output structural MRI file with corrected sform.

    Notes
    -----
    The matlab file is assumed to contain a variable called 'data' which is
    a matrix of size nSamples x nChannels.
    The matlab file is assumed to contain a variable called 'fs' which is
    the sampling frequency.
    The tsv file is assumed to contain a header row, and the following columns:
    name, type, bad, x, y, z, qx, qy, qz
    The x,y,z columns are the sensor locations in metres.
    The qx,qy,qz columns are the sensor orientations in metres.
    """
    # correct sform for smri
    sform_std_fixed = correct_mri(smri_file, smri_fixed_file)

    # Note that later in this function, we will also apply this sform to
    # the sensor coordinates and orientations.
    # This is because, with the OPM Notts data, coregistration on the sensor coordinates
    # has already been carried out, and so the sensor coordinates need to stay matching
    # the coordinates used in the MRI

    # Convert passed in OPM matlab file and tsv file to fif file

    # Load in chan info
    chan_info = pd.read_csv(tsv_file, header=None, skiprows=[0], sep="\t")

    sensor_names = chan_info.iloc[:, 0].to_numpy().T
    sensor_locs = chan_info.iloc[:, 4:7].to_numpy().T  # in metres
    sensor_oris = chan_info.iloc[:, 7:10].to_numpy().T
    sensor_bads = chan_info.iloc[:, 3].to_numpy().T

    # import pdb; pdb.set_trace()

    # Need to undo original sform on sensor locs and oris and then apply new sform
    smri = nib.load(smri_file)
    overall_xform = sform_std_fixed @ np.linalg.pinv(smri.header.get_sform())

    # This trans isn't really mri to head, it is mri to "mri_fixed", but mri_fixed is not available as an option
    overall_xform_trans = Transform("mri", "head", overall_xform)

    # Note sensor_locs are in metres, overall_xform_trans is in mm
    sensor_locs = apply_trans(overall_xform_trans, sensor_locs.T * 1000).T / 1000
    sensor_oris = apply_trans(overall_xform_trans, sensor_oris.T * 1000).T / 1000

    # Create fif file from mat file and chan_info

    Fs = scipy.io.loadmat(notts_opm_mat_file)["fs"][0, 0]  # Hz

    # see https://mne.tools/stable/auto_tutorials/simulation/10_array_objs.html

    info = mne.create_info(ch_names=sensor_names.tolist(), ch_types="mag", sfreq=Fs)

    # get names of bad channels
    select_indices = list(np.where(sensor_bads == "bad")[0])
    info["bads"] = sensor_names[select_indices].tolist()

    dig_montage = mne.channels.make_dig_montage(hsp=sensor_locs.T)
    info.set_montage(dig_montage)

    # MEG (device) -- dev_head_t --> HEAD (polhemus)
    # HEAD (polhemus)-- head_mri_t (polhemus2native) --> MRI (native)
    # MRI (native) -- mri_mrivoxel_t (native2nativeindex) --> MRI (native) voxel indices

    # We assume that device space, head space and mri space are all the same space
    # and that the sensor locations and fiducials (if there are any) are already in that space.
    # This means that dev_head_t is identity
    # This means that dev_mri_t is identity

    info["dev_head_t"] = Transform("meg", "head", np.identity(4))

    # Set sensor locs and oris in info

    def _cartesian_to_affine(loc, ori):

        # The orientation, ori, defines an orientation as a 3D cartesian vector (in x,y,z)
        # taken from the origin.
        # The location, loc, is a 3D cartesian vector (in x,y,z)
        # taken from the origin.

        # To convert the cartesian orientation vector to an affine rotation matrix, we first convert
        # the cartesian coordinates into spherical coords.
        # See https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates

        # r = 1
        theta = np.arccos(ori[2] / np.sqrt(np.sum(np.square(ori))))
        if ori[0] > 0:
            phi = np.arctan(ori[1] / ori[0])
        elif ori[0] < 0 and ori[1] >= 0:
            phi = np.arctan(ori[1] / ori[0]) + np.pi
        elif ori[0] < 0 and ori[1] < 0:
            phi = np.arctan(ori[1] / ori[0]) - np.pi
        elif ori[0] == 0 and ori[1] > 0:
            phi = np.pi / 2
        elif ori[0] == 0 and ori[1] < 0:
            phi = -np.pi / 2

        # We next convert the spherical coords into an affine rotation matrix.
        #
        # See "Rotation matrix from axis and angle" at https://en.wikipedia.org/wiki/Rotation_matrix
        # Plus see https://en.wikipedia.org/wiki/Spherical_coordinate_system#/media/File:3D_Spherical.svg
        # We will use the Physics convention for spherical coordinates
        #
        # MNE assumes that affine transform to determine sensor location/orientation
        # is applied to a unit vector along the z-axis
        #
        # First we do a rotation to the x-axis
        # i.e. rotation pi/2 around y-axis
        # i.e. axis of rotation (ux,uy,uz) = (0,1,0)
        deg = np.pi / 2
        Rdeg = np.array(
            [
                [np.cos(deg), 0, np.sin(deg), 0],
                [0, 1, 0, 0],
                [-np.sin(deg), 0, np.cos(deg), 0],
                [0, 0, 0, 1],
            ]
        )

        # Second we then do a rotation of phi around the z-axis
        # i.e. axis of rotation (ux,uy,uz) = (0,0,1)
        phin = phi
        Rphi = np.array(
            [
                [np.cos(phin), -np.sin(phin), 0, 0],
                [np.sin(phin), np.cos(phin), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        # Third we  do a rotation of -(pi/2-theta) around the
        # axis of rotation (ux,uy,uz) = (-np.sin(phi), np.cos(phi), 0)
        ux = -np.sin(phi)
        uy = np.cos(phi)
        thetan = -(np.pi / 2.0 - theta)
        Rtheta = np.array(
            [
                [
                    ux * ux * (1 - np.cos(thetan)) + np.cos(thetan),
                    ux * uy * (1 - np.cos(thetan)),
                    uy * np.sin(thetan),
                    0,
                ],
                [
                    ux * uy * (1 - np.cos(thetan)),
                    uy * uy * (1 - np.cos(thetan)) + np.cos(thetan),
                    -ux * np.sin(thetan),
                    0,
                ],
                [-uy * np.sin(thetan), ux * np.sin(thetan), np.cos(thetan), 0],
                [0, 0, 0, 1],
            ]
        )

        # We also want to combine the rotation matrix with the translation.
        # So, finally we do the translation
        translate = np.array(
            [[1, 0, 0, loc[0]], [0, 1, 0, loc[1]], [0, 0, 1, loc[2]], [0, 0, 0, 1]]
        )

        affine = translate @ Rtheta @ Rphi @ Rdeg

        return affine

    # test:
    # affine_from_loc_ori([0, 0, 0], [0, 1, 1]/np.sqrt(2))@[1, 0, 0, 1]

    for cc in range(len(info["chs"])):
        affine_mat = _cartesian_to_affine(sensor_locs[:, cc], sensor_oris[:, cc])
        info["chs"][cc]["loc"] = _coil_trans_to_loc(affine_mat)
        info["chs"][cc]["coil_type"] = FIFF.FIFFV_COIL_POINT_MAGNETOMETER

    # FINALLY put data and info together and save to fif_file
    data = scipy.io.loadmat(notts_opm_mat_file)["data"].T * 1e-15  # fT
    raw = mne.io.RawArray(data, info)
    raw.save(fif_file, overwrite=True)


def correct_mri(smri_file, smri_fixed_file):
    """Correct the sform in the structural MRI file.

    Parameters
    ----------
    smri_file : str
        The structural MRI file.
    smri_fixed_file : str
        The output structural MRI file with corrected sform.


    Returns
    -------
    sform_std : ndarray
        The new sform.

    Notes
    -----
    The sform is corrected so that it is in standard orientation.

    """
    # Copy smri_name to new file for modification
    copyfile(smri_file, smri_fixed_file)

    smri = nib.load(smri_fixed_file)
    sform = smri.header.get_sform()
    sform_std = np.copy(sform)

    # sform_std[0, 0:4] = [-1, 0, 0, 128]
    # sform_std[1, 0:4] = [0, 1, 0, -128]
    # sform_std[2, 0:4] = [0, 0, 1, -90]

    sform_std[0, 0:4] = [1, 0, 0, -90]
    sform_std[1, 0:4] = [0, -1, 0, 126]
    sform_std[2, 0:4] = [0, 0, -1, 72]

    os.system(
        "fslorient -setsform {} {}".format(
            " ".join(map(str, sform_std.flatten())), smri_fixed_file
        )
    )

    return sform_std
