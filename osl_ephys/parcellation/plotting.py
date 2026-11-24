"""Plots of parcellated data: spectra, correlations, source topographies and the QC plots."""

from __future__ import annotations

from pathlib import Path
import mne
import scipy
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.sparse.linalg
from scipy.signal import welch
from nilearn.plotting import plot_markers, plot_glass_brain
from mpl_toolkits.axes_grid1 import make_axes_locatable
from osl_ephys.parcellation import nii
from osl_ephys.files.functions import find_file
from osl_ephys.utils.logger import log_or_print
from osl_ephys.parcellation.parcellation import (
    guess_parcellation,
    parcel_centers,
    parcel_vector_to_nifti,
)


def plot_psds(
    parc_fif: str,
    parcellation_file: str,
    fmin: float = 0.5,
    fmax: float = 45,
    filename: str | None = None,
) -> None:
    """Plot PSD of each parcel time course.

    Parameters
    ----------
    parc_fif : mne.Raw or mne.Epochs
        MNE Raw or Epochs object containing the parcel data.
    parcellation_file : str
        Path to parcellation file.
    fmin : float, optional
        Minimum frequency.
    fmax : float, optional
        Maximum frequency.
    filename : str, optional
        Output filename.
    """
    if "epo.fif" in parc_fif:
        epochs = mne.read_epochs(parc_fif)
        fs = epochs.info["sfreq"]
        parc_ts = epochs.get_data(picks="misc")
    else:
        raw = mne.io.read_raw_fif(parc_fif)
        fs = raw.info["sfreq"]
        parc_ts = raw.get_data(picks="misc", reject_by_annotation="omit")

    if parc_ts.ndim == 3:
        # Calculate PSD for each epoch individually and average.
        # parc_ts shape is (n_epochs, n_channels, n_samples)
        psd = []
        for i in range(parc_ts.shape[0]):
            f, p = scipy.signal.welch(parc_ts[i], fs=fs, nperseg=fs, nfft=fs * 2)
            psd.append(p)
        psd = np.mean(psd, axis=0)
    else:
        # Calculate PSD of continuous data
        f, psd = scipy.signal.welch(parc_ts, fs=fs, nperseg=fs, nfft=fs * 2)

    # Plot
    from osl_ephys.utils.plotting import plot_psd_topo

    plot_psd_topo(
        f,
        psd,
        parcellation_file=parcellation_file,
        frequency_range=[fmin, fmax],
        filename=filename,
    )


def save_qc_plots(
    qc_dir: str | Path,
    id: str,
    parc_fif: str,
    parcellation_file: str,
    power_maps: bool = False,
    show: bool = False,
    cmap: str = "hot",
    summary: dict | None = None,
) -> None:
    """Save parcellation QC plots.

    Saves the following files to ``qc_dir/parc/id``, where the QC report
    looks for them:

    - ``psd_topo.webp``: PSD topography plot
    - ``power_maps.webp``: composite band power maps (only if
      ``power_maps=True``)

    Parameters
    ----------
    qc_dir : str or Path
        QC directory.
    id : str
        Session identifier.
    parc_fif : str
        Path to parcellated fif file.
    parcellation_file : str
        Parcellation file name.
    power_maps : bool, optional
        Whether to create band power map plots. Default is False.
    show : bool, optional
        Whether to display the plots interactively. Default is False.
    cmap : str, optional
        Colormap for power maps.
    summary : dict, optional
        Numbers to add to ``summary.json`` (e.g. the dipoles per parcel).
    """
    output_dir = Path(qc_dir) / "parc" / id
    output_dir.mkdir(parents=True, exist_ok=True)

    from osl_ephys.report import headless

    with headless():
        _save_qc_plots(
            output_dir, parc_fif, parcellation_file, power_maps, show, cmap, summary
        )


def _save_qc_plots(
    output_dir, parc_fif, parcellation_file, power_maps, show, cmap, summary
):
    from osl_ephys.report import save_summary
    from osl_ephys.utils import spectral as power
    from osl_ephys.utils.plotting import plot_psd_topo, plot_brain_surface

    # Load data and compute PSD once
    if "epo.fif" in parc_fif:
        parc_raw = mne.read_epochs(parc_fif)
        parc_ts = parc_raw.get_data(picks="misc")
    else:
        parc_raw = mne.io.read_raw_fif(parc_fif)
        parc_ts = parc_raw.get_data(picks="misc", reject_by_annotation="omit")

    fs = parc_raw.info["sfreq"]
    save_summary(
        output_dir,
        {
            "n_parcels": int(parc_ts.shape[-2]),
            "parcellation_file": str(parcellation_file),
            "sfreq": float(fs),
            **(summary or {}),
        },
    )
    f, psd = scipy.signal.welch(parc_ts, fs=fs, nperseg=fs, nfft=fs * 2)
    if psd.ndim == 3:
        # Average over epochs
        psd = psd.mean(axis=0)

    # PSD topography
    plot_psd_topo(
        f,
        psd,
        parcellation_file=parcellation_file,
        frequency_range=[1, 45],
        filename=str(output_dir / "psd_topo.webp"),
    )
    if not show:
        plt.close("all")

    if not power_maps:
        return

    # Band power maps — render each band and composite into a single image
    bands = {
        "delta": [1, 4],
        "theta": [4, 8],
        "alpha": [8, 13],
        "beta": [13, 30],
        "gamma": [30, 45],
    }
    band_images = []
    for band_name, freq_range in bands.items():
        band_power = power.variance_from_spectra(f, psd, frequency_range=freq_range)
        fig, ax = plot_brain_surface(
            band_power,
            parcellation_file=parcellation_file,
            title=f"{band_name} ({freq_range[0]}-{freq_range[1]} Hz)",
            cmap=cmap,
            symmetric_cbar=False,
        )
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        band_images.append(img)
        plt.close(fig)

    composite_fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    for ax, img in zip(axes, band_images):
        ax.imshow(img)
        ax.axis("off")
    composite_fig.tight_layout()
    composite_fig.savefig(
        str(output_dir / "power_maps.webp"),
        dpi=100,
        bbox_inches="tight",
        pil_kwargs={"quality": 80},
    )
    if not show:
        plt.close(composite_fig)


def plot_correlation(parc_ts, filename):
    """Plot correlation between parcel time courses.

    Parameters
    ----------
    parc_ts : np.ndarray
        (parcels, time) or (parcels, time, epochs) time series.
    filename : str
        Output filename.
    """
    if parc_ts.ndim == 3:
        # (parcels, time, epochs) -> (parcels, time)
        shape = parc_ts.shape
        parc_ts = parc_ts.reshape(shape[0], shape[1] * shape[2])

    # Calculate correlation
    corr = np.corrcoef(parc_ts)
    np.fill_diagonal(corr, 0)

    # Plot
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(corr)
    ax.set_xlabel("Parcel", fontsize=14)
    ax.set_ylabel("Parcel", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)
    fig.colorbar(im, cax=cax, orientation="vertical")

    # Save
    log_or_print(f"saving {filename}")
    plt.savefig(filename)
    plt.close(fig)


def plot_source_topo(
    data_map,
    parcellation_file=None,
    mask_file="MNI152_T1_8mm_brain.nii.gz",
    axis=None,
    cmap=None,
    vmin=None,
    vmax=None,
    alpha=0.7,
    freesurfer=False,
):
    """Plot a data map on a cortical surface. Wrapper for nilearn.plotting.plot_glass_brain.

    Parameters
    ----------
    data_map : array_like
        Vector of data values to plot (nparc,)
    parcellation_file : str
        Filepath of parcellation file to plot data on
    mask_file : str
        Filepath of mask file to plot data on (Default value = 'MNI152_T1_8mm_brain.nii.gz')
    axis : {None or axis handle}
        Axis to plot into (Default value = None)
    cmap : {None or matplotlib colormap}
        Colormap to use for plotting (Default value = None)
    vmin : {None or float}
        Minimum value for colormap (Default value = None)
    vmax : {None or float}
        Maximum value for colormap (Default value = None)
    alpha : {None or float}
        Alpha value for colormap (Default value = None)
    freesurfer : bool
        If True, parcellation_file is a freesurfer parcellation. Otherwise, it is a nifti file.

    Returns
    -------
    image : :py:class:`matplotlib.image.AxesImage <matplotlib.image.AxesImage>`
        AxesImage object
    """

    if parcellation_file is None:
        parcellation_file = guess_parcellation(data_map)
    parcellation_file = find_file(parcellation_file, freesurfer=freesurfer)
    mask_file = find_file(mask_file, freesurfer=freesurfer)

    if vmin is None:
        vmin = data_map.min()
    if vmax is None:
        vmax = data_map.max()

    if vmin < 0 and vmax > 0:
        vmax = np.max(np.abs([vmin, vmax]))
        vmin = -vmax

    if cmap is None:
        if vmin < 0 and vmax > 0:
            cmap = "RdBu_r"
        elif vmin >= 0:
            cmap = "Reds"
        else:
            cmap = "Blues_r"

    if axis is None:
        # Create figure
        fig, axis = plt.subplots()

    # Fill parcel values into a 3D voxel grid
    data_map = parcel_vector_to_nifti(data_map, parcellation_file).get_fdata()
    data_map = data_map[..., np.newaxis]
    mask = nib.load(mask_file)
    nii = nib.Nifti1Image(data_map, mask.affine, mask.header)

    # Plot
    plot_glass_brain(
        nii,
        output_file=None,
        display_mode="z",
        colorbar=False,
        axes=axis,
        cmap=cmap,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        plot_abs=False,
        annotate=False,
    )

    # despite the options of vmin, vmax, the colorbar is always set to -vmax to vmax. correct this
    # plt.gca().get_images()[0].set_clim(vmin, vmax)
    return plt.gca().get_images()[0]


def plot_psd(
    parc_ts, fs, parcellation_file, filename, freq_range=None, freesurfer=False
):
    """Plot PSD of each parcel time course.

    Parameters
    ----------
    parc_ts : np.ndarray
        (parcels, time) or (parcels, time, epochs) time series.
    fs : float
        Sampling frequency in Hz.
    parcellation_file : str
        Path to parcellation file.
    filename : str
        Output filename.
    freq_range : list of len 2
        Low and high frequency in Hz.
    freesurfer : bool
        Is the parcellation a FreeSurfer parcellation?
    """
    if parc_ts.ndim == 3:
        # Calculate PSD for each epoch individually and average
        psd = []
        for i in range(parc_ts.shape[-1]):
            f, p = welch(parc_ts[..., i], fs=fs, nperseg=fs, nfft=fs * 2)
            psd.append(p)
        psd = np.mean(psd, axis=0)
    else:
        # Calculate PSD of continuous data
        f, psd = welch(parc_ts, fs=fs, nperseg=fs, nfft=fs * 2)

    n_parcels = psd.shape[0]

    if freq_range is None:
        freq_range = [f[0], f[-1]]

    # Re-order to use colour to indicate anterior->posterior location
    parc_centers = parcel_centers(parcellation_file, freesurfer=freesurfer)
    order = np.argsort(parc_centers[:, 1])
    parc_centers = parc_centers[order]
    psd = psd[order]

    # Plot PSD
    fig, ax = plt.subplots()
    cmap = plt.cm.viridis_r
    for i in reversed(range(n_parcels)):
        ax.plot(f, psd[i], c=cmap(i / n_parcels))
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("PSD (a.u.)", fontsize=14)
    ax.set_xlim(freq_range[0], freq_range[1])
    ax.tick_params(axis="both", labelsize=14)
    plt.tight_layout()

    # Plot parcel topomap
    inside_ax = ax.inset_axes([0.45, 0.55, 0.5, 0.55])
    plot_markers(
        np.arange(n_parcels), parc_centers, node_size=12, colorbar=False, axes=inside_ax
    )

    # Save
    log_or_print(f"saving {filename}")
    plt.savefig(filename)
    plt.close()
