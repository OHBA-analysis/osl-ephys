"""Plots of parcel data: PSD topographies and brain surfaces."""

from __future__ import annotations

import logging

from typing import List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from nilearn.plotting import plot_markers, plot_img_on_surf

from osl_ephys import files
from osl_ephys.parcellation.parcellation import parcel_vector_to_nifti, Parcellation

_logger = logging.getLogger("osl_ephys")


def close(fig: Optional[plt.Figure] = None) -> None:
    """Close a figure.

    Parameters
    ----------
    fig : plt.figure, optional
        Figure to close. Defaults to all figures.
    """
    if fig is None:
        fig = "all"
    plt.close(fig)


def create_figure(*args, **kwargs):
    """Creates matplotlib figure and axes objects.

    Parameters
    ----------
    fig_kwargs
        Arguments to pass to `plt.subplots <https://matplotlib.org/stable/api\
        /_as_gen/matplotlib.pyplot.subplots.html>`_.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure.
    ax : array of plt.axes
        Array of axes (or single axis).
    """
    fig, ax = plt.subplots(*args, **kwargs)
    return fig, ax


def save(fig: plt.Figure, filename: str, tight_layout: bool = False) -> None:
    """Save and close a figure.

    Parameters
    ----------
    fig : plt.figure
        Matplotlib figure object.
    filename : str
        Output filename.
    tight_layout : bool, optional
        Should we call :code:`fig.tight_layout()`?
    """
    _logger.info(f"Saving {filename}")
    if tight_layout:
        fig.tight_layout()
    fig.savefig(filename)
    close(fig)


def plot_psd_topo(
    f: np.ndarray,
    psd: np.ndarray,
    only_show: Optional[list] = None,
    parcellation_file: Optional[str] = None,
    frequency_range: Optional[list] = None,
    topomap_pos: Optional[list] = None,
    cmap: str = "viridis",
    fig_kwargs: Optional[dict] = None,
    ax: Optional[plt.Axes] = None,
    filename: Optional[str] = None,
):
    """Plot PSDs for parcels and a topomap.

    Parameters
    ----------
    f : np.ndarray
        Frequency axis. Shape must be (n_freq,).
    psd : np.ndarray
        PSD for each parcel. Shape must be (n_parcels, n_freq).
    only_show : list, optional
        Indices for parcels to include in the plot. Defaults to all parcels.
    parcellation_file : str, optional
        Path to parcellation file.
    frequency_range : list, optional
        Min and max frequency for the x-axis.
    topomap_pos : list, optional
        Positioning and size of the topomap: :code:`[x0, y0, width, height]`.
        :code:`x0`, :code:`y0`, :code:`width`, :code:`height` should be floats
        between 0 and 1. Defaults to :code:`[0.45, 0.55, 0.5, 0.55]` to place
        the topomap on the top right. This is not used if
        :code:`parcellation_file=None`.
    cmap : str, optional
        Matplotlib colormap.
    fig_kwargs : dict, optional
        Arguments to pass to :code:`plt.subplots()`.
    ax : matplotlib Axis, optional
        Axis to plot on.
    filename : str, optional
        Output filename.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`ax=None` and
        :code:`filename=None`.
    ax : plt.axes
        Matplotlib axis object(s). Only returned if :code:`ax=None` and
        :code:`filename=None`.
    """
    if frequency_range is None:
        frequency_range = [f[0], f[-1]]

    if topomap_pos is None:
        topomap_pos = [0.45, 0.55, 0.5, 0.55]

    if fig_kwargs is None:
        fig_kwargs = {}

    if parcellation_file is not None:
        # Get the center of each parcel
        parcellation = Parcellation(parcellation_file)
        roi_centers = parcellation.roi_centers()

        # Re-order to use colour to indicate anterior->posterior location
        order = np.argsort(roi_centers[:, 1])
        roi_centers = roi_centers[order]
        psd = np.copy(psd)[order]

    n_parcels = psd.shape[0]

    # Which parcels should we plot the PSD for?
    if only_show is None:
        only_show = np.arange(n_parcels)
    elif parcellation_file is not None:
        # The PSDs have been re-ordered (anterior -> posterior), so we
        # need to map the parcel indices to their re-ordered positions
        only_show = np.argsort(order)[np.asarray(only_show)]

    # Create axis
    if ax is None:
        fig, ax = create_figure(**fig_kwargs)
        ax_passed = False
    else:
        ax_passed = True

    # Plot PSDs
    cmap = plt.get_cmap(cmap + "_r")
    for i in reversed(range(n_parcels)):
        if i in only_show:
            ax.plot(f, psd[i], c=cmap(i / n_parcels))
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (a.u.)")
    ax.set_xlim(frequency_range[0], frequency_range[-1])
    plt.tight_layout()

    if parcellation_file is not None:
        # Plot parcel topomap
        inside_ax = ax.inset_axes(topomap_pos)
        plot_markers(
            np.arange(parcellation.n_parcels),
            roi_centers,
            node_size=12,
            node_cmap=cmap,
            colorbar=False,
            axes=inside_ax,
        )

    # Save
    if filename is not None and not ax_passed:
        save(fig, filename)
    elif not ax_passed:
        return fig, ax


def plot_brain_surface(
    values: np.ndarray,
    parcellation_file: str,
    title: Optional[str] = None,
    cmap: Union[str, matplotlib.colors.ListedColormap] = "cold_hot",
    colorbar: bool = True,
    symmetric_cbar: bool = True,
    cbar_tick_format: str = "%.2g",
    cbar_fontsize: int = 24,
    cbar_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    hemispheres: Optional[list] = None,
    views: Optional[list] = None,
    bg_on_data: bool = False,
    threshold: Optional[float] = None,
    remove_subcortical_voxels: bool = False,
    filename: Optional[str] = None,
    show_plot: Optional[bool] = None,
):
    """Plot a 2D heat map on the surface of the brain.

    Parameters
    ----------
    values : np.ndarray
        Data to plot. Must be of shape (n_parcels,).
    parcellation_file : str
        Parcellation file. See osl_ephys.files.parcellation.
    title : str, optional
        Title for the plot.
    cmap : str or matplotlib.colors.ListedColormap, optional
        Matplotlib colormap.
    colorbar : bool, optional
        Should we plot a colorbar?
    symmetric_cbar : bool, optional
        Should we have a symmetric color bar?
    cbar_tick_format : str, optional
        Formatting for the color bar tick labels.
        Example use: :code:`cbar_tick_format='%.2f'`.
    cbar_fontsize : int, optional
        Fontsize for the color bar ticks and label.
    cbar_label : str, optional
        Label for the color bar.
    vmin : float, optional
        Minimum value for the color bar.
        May be overridden if :code:`symmetric_cbar=True`.
    vmax : float, optional
        Maximum value for the color bar.
        May be overridden if :code:`symmetric_cbar=True`.
    hemispheres : list, optional
        :code:`['left', 'right']` or :code:`['left']` or :code:`['right']`.
        Defaults to :code:`['left', 'right']`.
    views : list, optional
        The list can contain :code:`'lateral'` or :code:`'medial'`.
        Defaults to :code:`['lateral']`, which will show one row with
        lateral views. :code:`['lateral', 'medial']` will show two
        rows with the lateral view on top and medial view below.
    bg_on_data : bool, optional
        If True, the sulcal depth is jointly visible with surface data.
        Otherwise, the background image will only be visible where there
        is no surface data (either because the surface data contains nans
        or because is was thresholded).
    threshold : float, optional
        Threshold values to display. Defaults to no thresholding.
    remove_subcortical_voxels : bool, optional
        Should we set the subcortical voxels to np.nan?
    filename : str, optional
        Output filename. Extension can be :code:`png/svg/pdf`.
        If None is passed then the image is shown on screen and the
        Matplotlib objects are returned.
    show_plot : bool, optional
        Should we show the plot? If :code:`filename` is True, defaults
        to False, otherwise False.
    mask_file : str, optional
        Deprecated and ignored: brain maps are plotted on the voxel grid of
        the parcellation file. Will be removed in v3.8.0.

    Returns
    -------
    fig : plt.figure
        Matplotlib figure object. Only returned if :code:`filename` is None.
    ax : plt.axes
        Matplotlib axis object. Only returned if :code:`filename` is None.
    """
    if vmin is None:
        vmin = np.min(values)
    if vmax is None:
        vmax = np.max(values)
    if symmetric_cbar:
        vmax = np.max([vmax, -vmin])
        vmin = -vmax
    if hemispheres is None:
        hemispheres = ["left", "right"]
    if views is None:
        views = ["lateral"]

    if filename is not None:
        allowed_extensions = [".png", ".svg", ".pdf"]
        if not any([ext in filename for ext in allowed_extensions]):
            raise ValueError(
                "filename must have one of following extensions: "
                f"{' '.join(allowed_extensions)}."
            )

    if filename is None:
        show_plot = True
    else:
        show_plot = False

    # Convert from parcel values to voxel values
    nii = parcel_vector_to_nifti(values, parcellation_file, remove_subcortical_voxels)

    # Plot
    fig, ax = plot_img_on_surf(
        nii,
        output_file=None,
        colorbar=False,
        cmap=cmap,
        symmetric_cbar=symmetric_cbar,
        vmin=vmin,
        vmax=vmax,
        hemispheres=hemispheres,
        views=views,
        bg_on_data=bg_on_data,
        threshold=threshold,
    )

    if views == ["lateral"]:
        # Plotting 2 views

        # Title
        fig.suptitle(title, fontsize=30, y=0.97)

        if colorbar:
            # Positioning and size of the colour bar
            # [left, bottom, width, height]
            cbar_ax = fig.add_axes([0.25, 0.2, 0.5, 0.05])

            # Colour bar
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            cbar = fig.colorbar(
                sm,
                cax=cbar_ax,
                orientation="horizontal",
                format=cbar_tick_format,
            )
            cbar.ax.tick_params(labelsize=cbar_fontsize)
            cbar.set_label(cbar_label, fontsize=cbar_fontsize)
    else:
        # Plotting 4 views

        # Title
        fig.suptitle(title, fontsize=22, y=0.98)

        if colorbar:
            # Positioning and size of the colour bar
            # [left, bottom, width, height]
            cbar_ax = fig.add_axes([0.3, 0.1, 0.4, 0.04])

            # Colour bar
            sm = plt.cm.ScalarMappable(
                cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            cbar = fig.colorbar(
                sm,
                cax=cbar_ax,
                orientation="horizontal",
                format=cbar_tick_format,
            )
            cbar.ax.tick_params(labelsize=16)
            cbar.set_label(cbar_label, fontsize=16)

    # Save or return
    if filename is not None:
        fig.savefig(filename)
        if not show_plot:
            plt.close(fig)
        return fig
    else:
        return fig, ax
