"""Plotting functions for ICA.

"""

# Authors: Mats van Es <mats.vanes@psych.ox.ac.uk>
import logging
import numpy as np
import matplotlib.pyplot as plt
from mne._fiff.pick import channel_type

import matplotlib

# Force matplotlib to use an interactive backend:
backends_to_try = ['Qt5Agg', 'QtAgg', 'ipympl', 'GTK3Agg', 'GTK4Agg', 'macosx', 'TkAgg', 'GTK3Cairo', 'GTK4Cairo', 'wxAgg' ]
for backend in backends_to_try:
    try:
        matplotlib.use(backend, force=True)
        break
    except ImportError:
        continue


# Configure logging
logger = logging.getLogger(__name__)

def plot_ica(
    ica,
    inst,
    picks=None,
    start=None,
    stop=None,
    title=None,
    show=True,
    block=False,
    show_first_samp=False,
    show_scrollbars=True,
    time_format="float",
    n_channels=10,
    bad_labels_list=["eog", "ecg", "emg", "hardware", "other"],
):
    """osl-ephys' adaptation of MNE's :py:meth:`mne.preprocessing.ICA.plot_sources <mne.preprocessing.ICA.plot_sources>` function to 
    plot estimated latent sources given the unmixing matrix.

    Typical usecases:

    1. plot evolution of latent sources over time based on (Raw input)
    2. plot latent source around event related time windows (Epochs input)
    3. plot time-locking in ICA space (Evoked input)

    Parameters
    ----------
    ica : :py:class:`mne.preprocessing.ICA <mne.preprocessing.ICA>`.
        The ICA solution.
    inst : :py:class:`mne.io.Raw <mne.io.Raw>`, :py:class:`mne.Epochs <mne.Epochs>`, or :py:class:`mne.Evoked <mne.Evoked>`.
        The object to plot the sources from.
    picks : str
        Channel types to pick.
    start, stop : float | int | None
        If ``inst`` is a :py:class:`mne.io.Raw <mne.io.Raw>` or an  :py:class:`mne.Evoked <mne.Evoked>` object, the first and
        last time point (in seconds) of the data to plot. If ``inst`` is a
        :py:class:`mne.io.Raw <mne.io.Raw>` object, ``start=None`` and ``stop=None`` will be
        translated into ``start=0.`` and ``stop=3.``, respectively. For
        :py:class:`mne.Evoked <mne.Evoked>`, ``None`` refers to the beginning and end of the evoked
        signal. If ``inst`` is an  :py:class:`mne.Epochs <mne.Epochs>` object, specifies the index of
        the first and last epoch to show.
    title : str | None
        The window title. If None a default is provided.
    show : bool
        Show figure if True.
    block : bool
        Whether to halt program execution until the figure is closed.
        Useful for interactive selection of components in raw and epoch
        plotter. For evoked, this parameter has no effect. Defaults to False.
    show_first_samp : bool
        If True, show time axis relative to the ``raw.first_samp``.
    n_channels : int
        Number of channels to show at the same time (default: 10)
    bad_labels_list : list of str
        list of bad labels to show in the bad labels list that can be used to mark the type of 
        bad component. Defaults to ``["eog", "ecg", "emg", "hardware", "other"]``.
 

    Returns
    -------
    fig : instance of Figure
        The figure.

    Notes
    -----
    For raw and epoch instances, it is possible to select components for
    exclusion by clicking on the line. The selected components are added to
    ``ica.exclude`` on close.

    .. versionadded:: 0.10.0
    """
    from mne.io.base import BaseRaw
    from mne.io.pick import _picks_to_idx  # OSL ADDITION
    from mne.evoked import Evoked
    from mne.epochs import BaseEpochs

    exclude = ica.exclude
    picks = _picks_to_idx(ica.n_components_, picks, "all")

    if isinstance(inst, (BaseRaw, BaseEpochs)):
        fig = _plot_sources(
            ica,
            inst,
            picks,
            exclude,
            start=start,
            stop=stop,  # OSL VERSION
            show=show,
            title=title,
            block=block,
            show_first_samp=show_first_samp,
            show_scrollbars=show_scrollbars,
            time_format=time_format,
            n_channels=n_channels,
            bad_labels_list=bad_labels_list,
        )
    elif isinstance(inst, Evoked):
        if start is not None or stop is not None:
            inst = inst.copy().crop(start, stop)
        sources = ica.get_sources(inst)
        fig = _plot_ica_sources_evoked(
            evoked=sources,
            picks=picks,
            exclude=exclude,
            title=title,
            labels=getattr(ica, "labels_", None),
            show=show,
            ica=ica,
            n_channels=n_channels,
            bad_labels_list=bad_labels_list,
        )
    else:
        raise ValueError("Data input must be of Raw or Epochs type")

    return fig


def _plot_sources(
    ica,
    inst,
    picks,
    exclude,
    start,
    stop,
    show,
    title,
    block,
    show_scrollbars,
    show_first_samp,
    time_format,
    n_channels,
    bad_labels_list,
):
    """Adaptation of MNE's `mne.preprocessing.ica._plot_sources` function to allow for OSL additions.
    """    
    """Plot the ICA components as a RawArray or EpochsArray."""
    # from mne.viz._figure import _get_browser
    from mne.viz.utils import _compute_scalings, _make_event_color_dict, plt_show
    from mne import EpochsArray, BaseEpochs
    from mne.io import RawArray, BaseRaw
    from mne import create_info
    from mne import pick_types
    from mne.defaults import _handle_default

    # handle defaults / check arg validity
    is_raw = isinstance(inst, BaseRaw)
    is_epo = isinstance(inst, BaseEpochs)
    sfreq = inst.info["sfreq"]
    color = _handle_default("color", (0.0, 0.0, 0.0))
    units = _handle_default("units", None)
    scalings = (
        _compute_scalings(None, inst)
        if is_raw
        else _handle_default("scalings_plot_raw")
    )
    scalings["misc"] = 5.0
    scalings["whitened"] = 1.0
    unit_scalings = _handle_default("scalings", None)

    # data
    if is_raw:
        data = ica._transform_raw(inst, 0, len(inst.times))[picks]
    else:
        data = ica._transform_epochs(inst, concatenate=True)[picks]

    # events
    if is_epo:
        event_id_rev = {v: k for k, v in inst.event_id.items()}
        event_nums = inst.events[:, 2]
        event_color_dict = _make_event_color_dict(None, inst.events, inst.event_id)

    # channel properties / trace order / picks
    ch_names = list(ica._ica_names)  # copy
    ch_types = ["misc" for _ in picks]

    # add EOG/ECG channels if present
    eog_chs = pick_types(inst.info, meg=False, eog=True, ref_meg=False)
    extra_picks = pick_types(inst.info, meg=False, ecg=True, eog=True, ref_meg=False)
    for idx in extra_picks[::-1]:
        ch_names.insert(0, inst.ch_names[idx])
        ch_types.insert(0, "eog" if idx in eog_chs else "ecg")
    if len(extra_picks):
        if is_raw:
            eog_ecg_data, _ = inst[extra_picks, :]
        else:
            eog_ecg_data = np.concatenate(inst.get_data(extra_picks), axis=1)
        data = np.append(eog_ecg_data, data, axis=0)
    picks = np.concatenate((picks, ica.n_components_ + np.arange(len(extra_picks))))
    ch_order = np.arange(len(picks))
    n_channels = min([n_channels, len(picks)])
    ch_names_picked = [ch_names[x] for x in picks]
    
    # because we added channels to the beginning of the data, we need to adjust exclude:
    exclude = [x + len(extra_picks) for x in exclude if x in picks]
    
    # create info
    info = create_info(ch_names_picked, sfreq, ch_types=ch_types)
    with info._unlock():
        info["meas_date"] = inst.info["meas_date"]
    info["bads"] = [ch_names[x] for x in exclude if x in picks]
    if is_raw:
        inst_array = RawArray(data, info, inst.first_samp)
        inst_array.set_annotations(inst.annotations)
    else:
        data = data.reshape(-1, len(inst), len(inst.times)).swapaxes(0, 1)
        inst_array = EpochsArray(data, info)

    # handle time dimension
    start = 0 if start is None else start
    _last = inst.times[-1] if is_raw else len(inst.events)
    stop = min(start + 20, _last) if stop is None else stop
    first_time = inst._first_time if show_first_samp else 0
    if is_raw:
        duration = stop - start
        start += first_time
    else:
        n_epochs = stop - start
        total_epochs = len(inst)
        epoch_n_times = len(inst.times)
        n_epochs = min(n_epochs, total_epochs)
        n_times = total_epochs * epoch_n_times
        duration = n_epochs * epoch_n_times / sfreq
        event_times = (
            np.arange(total_epochs) * epoch_n_times + inst.time_as_index(0)
        ) / sfreq
        # NB: this includes start and end of data:
        boundary_times = np.arange(total_epochs + 1) * epoch_n_times / sfreq
    if duration <= 0:
        raise RuntimeError("Stop must be larger than start.")

    # misc
    bad_color = "lightgray"
    title = "ICA components" if title is None else title
    
    # OSL ADDITION
    # define some colors for bad component labels
    import matplotlib.colors as mcolors

    c = list(mcolors.TABLEAU_COLORS.keys())
    idx = [c.index(i) for i in c if "red" in i]
    for i in idx:
        del c[i]
    c = c[: len(bad_labels_list) + 1]  # keep as many as required.

    params = dict(
        inst=inst_array,
        ica=ica,
        ica_inst=inst,
        info=info,
        # channels and channel order
        ch_names=np.array(ch_names_picked),
        ch_types=np.array(ch_types),
        ch_order=ch_order,
        picks=picks,
        n_channels=n_channels,
        picks_data=list(),
        bad_labels_list=bad_labels_list,  # OSL ADDITION
        # time
        t_start=start if is_raw else boundary_times[start],
        duration=duration,
        n_times=inst.n_times if is_raw else n_times,
        first_time=first_time,
        time_format=time_format,
        decim=1,
        # events
        event_times=None if is_raw else event_times,
        # preprocessing
        projs=list(),
        projs_on=np.array([], dtype=bool),
        apply_proj=False,
        remove_dc=True,  # for EOG/ECG
        filter_coefs=None,
        filter_bounds=None,
        noise_cov=None,
        # scalings
        scalings=scalings,
        units=units,
        unit_scalings=unit_scalings,
        # colors
        ch_color_bad=bad_color,
        ch_color_dict=color,
        bad_label_colors=c,
        # display
        butterfly=False,
        clipping=None,
        scrollbars_visible=show_scrollbars,
        scalebars_visible=False,
        window_title=title,
    )
    if is_epo:
        params.update(
            n_epochs=n_epochs,
            boundary_times=boundary_times,
            event_id_rev=event_id_rev,
            event_color_dict=event_color_dict,
            event_nums=event_nums,
            epoch_color_bad=(1, 0, 0),
            epoch_colors=None,
            xlabel="Epoch number",
        )

    fig = _get_browser(**params)
    fig.mne.ch_start = len(extra_picks) # this is necessary to make sure to plot the EOG/ECG only once
    fig._update_picks()

    # update data, and plot
    fig._update_trace_offsets()
    fig._update_data()
    fig._draw_traces()  # OSL VERSION

    # plot annotations (if any)
    if is_raw:
        fig._setup_annotation_colors()
        fig._update_annotation_segments()
        fig._draw_annotations()

    plt_show(show, block=block)
    return fig


from mne.viz._mpl_figure import MNEBrowseFigure

backend = None


def _get_browser(**kwargs):
    """OSL Adaptation of MNE's `mne.viz._figure._get_browser` function 
    that instantiate a new MNE browse-style figure.

    """    
    from mne.viz.utils import _get_figsize_from_config
    from mne.viz._figure import _init_browser_backend

    figsize = kwargs.setdefault("figsize", _get_figsize_from_config())
    if figsize is None or np.any(np.array(figsize) < 8):
        kwargs["figsize"] = (8, 8)
    # Initialize browser backend
    _init_browser_backend()

    # Initialize Browser
    browser = _init_browser(
        backend, **kwargs
    )  # OSL ADDITION IN ORDER TO USE OSL'S FIGURE CLASS FROM _INIT_BROWSER

    return browser


def _init_browser(backend, **kwargs):  # OSL ADDITION IN ORDER TO USE OSL'S FIGURE CLASS
    from mne.viz._mpl_figure import _figure
    """OSL's adaptation of MNE's `mne.viz._mpl_figure._init_browser` that 
    instantiate a new MNE browse-style figure.
    """
    
    fig = _figure(toolbar=False, FigureClass=osl_MNEBrowseFigure, **kwargs)

    # initialize zen mode
    # (can't do in __init__ due to get_position() calls)
    fig.canvas.draw()
    fig._update_zen_mode_offsets()
    fig._resize(None)  # needed for MPL >=3.4
    # if scrollbars are supposed to start hidden,
    # set to True and then toggle
    if not fig.mne.scrollbars_visible:
        fig.mne.scrollbars_visible = True
        fig._toggle_scrollbars()

    return fig


class osl_MNEBrowseFigure(MNEBrowseFigure):
    """OSL's adaptatation of MNE's `mne.viz._mpl_figure.MNEBrowseFigure` that
    creates an interactive figure with scrollbars, for data browsing."""
    

    def __init__(self, inst, figsize, ica=None,
                 xlabel='Time (s)', **kwargs):
        from matplotlib.colors import to_rgba_array
        from matplotlib.patches import Rectangle
        from matplotlib.ticker import (FixedFormatter, FixedLocator,
                                       FuncFormatter, NullFormatter)
        from matplotlib.transforms import blended_transform_factory
        from matplotlib.widgets import Button
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from mpl_toolkits.axes_grid1.axes_size import Fixed

        # # OSL IMPORTS
        from mne import pick_types
        from mne import BaseEpochs
        from mne.io import BaseRaw
        from mne.preprocessing import ICA
        from mne.viz._figure import BrowserBase
        from mne.viz._mpl_figure import MNEFigure, _patched_canvas
        import mne
        from functools import partial

        self.backend_name = "matplotlib"

        kwargs.update({"inst": inst, "figsize": figsize, "ica": ica, "xlabel": xlabel})

        BrowserBase.__init__(self, **kwargs)
        MNEFigure.__init__(self, **kwargs)

        # MAIN AXES: default sizes (inches)
        # XXX simpler with constrained_layout? (when it's no longer "beta")
        l_margin = 0.8#1.0
        r_margin = 1.0#0.1
        b_margin = 0.45
        t_margin = 0.35
        scroll_width = 0.25
        hscroll_dist = 0.25
        vscroll_dist = 0.1
        help_width = scroll_width * 2
        # MVE: ADD SIZES FOR TOPOS
        extra_chans = pick_types(inst.info, meg=False, eeg=False, ref_meg=False, eog=True, ecg=True, exclude=[])
        exist_meg = any(ct in np.unique(ica.get_channel_types()) for ct in ['mag', 'grad'])
        exist_eeg = 'eeg' in np.unique(ica.get_channel_types())
        n_topos = len(
            np.unique(
                [
                    channel_type(ica.info, ch)
                    for ch in mne.pick_types(ica.info, meg=exist_meg, eeg=exist_eeg)
                ]
            )
        )
        topo_width_ratio = 8 + n_topos  # 1
        topo_dist = self._inch_to_rel(0.05)  # 0.25

        # MAIN AXES: default margins (figure-relative coordinates)
        # self.canvas.figure.clear() # clear axes (inherited from MNE) # TODO: Do we need this?
        left = self._inch_to_rel(l_margin - vscroll_dist - help_width)
        right = 1 - self._inch_to_rel(r_margin)
        bottom = self._inch_to_rel(b_margin, horiz=False)
        top = 1 - self._inch_to_rel(t_margin, horiz=False)
        height = top - bottom

        # OSL ADDITION: ADAPT SIZES OF TIME COURSE SUBPLOT AND ADD TOPO PLOT SIZE
        fullwidth = right - left
        width = (topo_width_ratio - n_topos) * (
            fullwidth - n_topos * topo_dist
        ) / topo_width_ratio - (
            self._inch_to_rel(hscroll_dist) + self._inch_to_rel(scroll_width)
        )  # width = right - left
        topo_width = (fullwidth - topo_dist) / topo_width_ratio
        topo_height = (
            height - self._inch_to_rel(hscroll_dist + b_margin)
        ) / self.mne.n_channels - topo_dist
        position = [
            left + n_topos * (topo_width + topo_dist),
            bottom,
            width,
            height,
        ]  # position = [left, bottom, width, height]
        # Main axes must be a subplot for subplots_adjust to work (so user can
        # adjust margins). That's why we don't use the Divider class directly.
        ax_main = self.add_axes(
            position
        )  # OSL ADDITION USE ADD_AXES INSTEAD OF ADD_SUBPLOT
        # OSL ADDITION: CREATE TOPO AXES
        ax_topo = np.empty((n_topos, self.mne.n_channels), dtype=object)
        for i in np.arange(n_topos):
            for j in np.arange(self.mne.n_channels):
                topo_position = [
                    left + i * (topo_width + topo_dist),
                    bottom
                    + ((self.mne.n_channels) - j) * (topo_height + topo_dist)*1.03
                    - self._inch_to_rel(0.13),
                    topo_width,
                    topo_height,
                ]
                ax_topo[i, j] = self.add_axes(topo_position)
                ax_topo[i, j].set_axis_off()
        self.subplotpars.update(left=left, bottom=bottom, top=top, right=right)
        div = make_axes_locatable(ax_main)
        # this only gets shown in zen mode
        self.mne.zen_xlabel = ax_main.set_xlabel(xlabel)
        self.mne.zen_xlabel.set_visible(not self.mne.scrollbars_visible)
        
        # make sure background color of the axis is set
        if 'bgcolor' in kwargs:
            ax_main.set_facecolor(kwargs['bgcolor'])

        # OSL ADDITION: GET POSITIONS FOR BAD LABELS LIST
        self.mne.bad_labels_xpos = 1 - self._inch_to_rel(r_margin + 0.35)
        self.mne.bad_labels_ypos = []
        for i in range(len(self.mne.bad_labels_list)+2):
            self.mne.bad_labels_ypos.append(1 - self._inch_to_rel(t_margin + 0.5 + 0.3*(i+1), horiz=False))

        # SCROLLBARS
        ax_hscroll = div.append_axes(
            position="bottom", size=Fixed(scroll_width), pad=Fixed(hscroll_dist)
        )
        ax_vscroll = div.append_axes(
            position="right", size=Fixed(scroll_width), pad=Fixed(vscroll_dist)
        )
        ax_hscroll.get_yaxis().set_visible(False)
        ax_hscroll.set_xlabel(xlabel)
        ax_vscroll.set_axis_off()
        # HORIZONTAL SCROLLBAR PATCHES (FOR MARKING BAD EPOCHS)
        if self.mne.is_epochs:
            epoch_nums = self.mne.inst.selection
            for ix, _ in enumerate(epoch_nums):
                start = self.mne.boundary_times[ix]
                width = np.diff(self.mne.boundary_times[:2])[0]
                ax_hscroll.add_patch(
                    Rectangle(
                        (start, 0), width, 1, color="none",
                        zorder=self.mne.zorder["patch"]))
            # both axes, major ticks: gridlines
            for _ax in (ax_main, ax_hscroll):
                _ax.xaxis.set_major_locator(FixedLocator(self.mne.boundary_times[1:-1]))
                _ax.xaxis.set_major_formatter(NullFormatter())
            grid_kwargs = dict(
                color=self.mne.fgcolor, axis="x", zorder=self.mne.zorder["grid"]
            )
            ax_main.grid(linewidth=2, linestyle="dashed", **grid_kwargs)
            ax_hscroll.grid(alpha=0.5, linewidth=0.5, linestyle="solid", **grid_kwargs)
            # main axes, minor ticks: ticklabel (epoch number) for every epoch
            ax_main.xaxis.set_minor_locator(FixedLocator(self.mne.midpoints))
            ax_main.xaxis.set_minor_formatter(FixedFormatter(epoch_nums))
            # hscroll axes, minor ticks: up to 20 ticklabels (epoch numbers)
            ax_hscroll.xaxis.set_minor_locator(
                FixedLocator(self.mne.midpoints, nbins=20)
            )
            ax_hscroll.xaxis.set_minor_formatter(
                FuncFormatter(lambda x, pos: self._get_epoch_num_from_time(x))
            )
            # hide some ticks
            ax_main.tick_params(axis="x", which="major", bottom=False)
            ax_hscroll.tick_params(axis="x", which="both", bottom=False)
        else:
            # RAW / ICA X-AXIS TICK & LABEL FORMATTING # TODO: OSL NOT SURE IF THIS BREAKS WITH PLOTTING FUNCTING
            ax_main.xaxis.set_major_formatter(
                FuncFormatter(partial(self._xtick_formatter, ax_type="main"))
            )
            ax_hscroll.xaxis.set_major_formatter(
                FuncFormatter(partial(self._xtick_formatter, ax_type="hscroll"))
            )
            if self.mne.time_format != "float":
                for _ax in (ax_main, ax_hscroll):
                    _ax.set_xlabel("Time (HH:MM:SS)")

        # VERTICAL SCROLLBAR PATCHES (COLORED BY CHANNEL TYPE)
        ch_order = self.mne.ch_order
        for ix, pick in enumerate(ch_order[len(extra_chans):]):
            this_color = (
                self.mne.ch_color_bad
                if self.mne.ch_names[pick] in self.mne.info["bads"]
                else self.mne.ch_color_dict
            )
            if isinstance(this_color, dict):
                this_color = this_color[self.mne.ch_types[pick]]
            ax_vscroll.add_patch(
                Rectangle(
                    (0, ix), 1, 1, color=this_color, zorder=self.mne.zorder["patch"]
                )
            )
        ax_vscroll.set_ylim(len(ch_order) - len(extra_chans), 0)
        ax_vscroll.set_visible(not self.mne.butterfly)
        # SCROLLBAR VISIBLE SELECTION PATCHES
        sel_kwargs = dict(
            alpha=0.3, linewidth=4, clip_on=False, edgecolor=self.mne.fgcolor
        )
        vsel_patch = Rectangle(
            (0, 0), 1, self.mne.n_channels - len(extra_chans), facecolor=self.mne.bgcolor, **sel_kwargs
        )
        ax_vscroll.add_patch(vsel_patch)
        hsel_facecolor = np.average(
            np.vstack(
                (to_rgba_array(self.mne.fgcolor), to_rgba_array(self.mne.bgcolor))
            ),
            axis=0,
            weights=(3, 1),
        )  # 75% foreground, 25% background
        hsel_patch = Rectangle(
            (self.mne.t_start, 0),
            self.mne.duration,
            1,
            facecolor=hsel_facecolor,
            **sel_kwargs,
        )
        ax_hscroll.add_patch(hsel_patch)
        ax_hscroll.set_xlim(
            self.mne.first_time,
            self.mne.first_time + self.mne.n_times / self.mne.info["sfreq"],
        )
        # VLINE
        vline_color = (0.0, 0.75, 0.0)
        vline_kwargs = dict(
            visible=False, animated=True, zorder=self.mne.zorder["vline"]
        )
        if self.mne.is_epochs:
            x = np.arange(self.mne.n_epochs)
            vline = ax_main.vlines(x, 0, 1, colors=vline_color, **vline_kwargs)
            vline.set_transform(
                blended_transform_factory(ax_main.transData, ax_main.transAxes)
            )
            vline_hscroll = None
        else:
            vline = ax_main.axvline(0, color=vline_color, **vline_kwargs)
            vline_hscroll = ax_hscroll.axvline(0, color=vline_color, **vline_kwargs)
        vline_text = ax_hscroll.text(
            self.mne.first_time,
            1.2,
            "",
            fontsize=10,
            ha="right",
            va="bottom",
            color=vline_color,
            **vline_kwargs,
        )

        # HELP BUTTON: initialize in the wrong spot...
        ax_help = div.append_axes(
            position="left", size=Fixed(help_width), pad=Fixed(vscroll_dist)
        )
        # HELP BUTTON: ...move it down by changing its locator
        loc = div.new_locator(nx=0, ny=0)
        ax_help.set_axes_locator(loc)
        # HELP BUTTON: make it a proper button
        with _patched_canvas(ax_help.figure):
            self.mne.button_help = Button(ax_help, "Help")
        # PROJ BUTTON
        ax_proj = None
        if len(self.mne.projs) and not inst.proj:
            proj_button_pos = [
                1 - self._inch_to_rel(r_margin + scroll_width),  # left
                self._inch_to_rel(b_margin, horiz=False),  # bottom
                self._inch_to_rel(scroll_width),  # width
                self._inch_to_rel(scroll_width, horiz=False),  # height
            ]
            loc = div.new_locator(nx=4, ny=0)
            ax_proj = self.add_axes(proj_button_pos)
            ax_proj.set_axes_locator(loc)
            with _patched_canvas(ax_help.figure):
                self.mne.button_proj = Button(ax_proj, "Prj")

        # INIT TRACES
        self.mne.trace_kwargs = dict(antialiased=True, linewidth=0.5)
        self.mne.traces = ax_main.plot(
            np.full((1, self.mne.n_channels), np.nan), **self.mne.trace_kwargs
        )

        # MVE: INITIALLY THIS IS WHERE I INITIALIZED THE TOPOS. TURNS OUT ITS REDUNDANT BECAUSE IT IS TAKEN CARE OF IN
        # THE INTERACTIVE PART OF THE FIGURE. IT ALSO SOLVES THE EXTRA BONUS FIGURE
        # INIT TOPOS
        # NOTE: Commenting the next line out seems to not break the code, but to solve the bonus figure that is created
        # upon running the code.
        # self.plot_topos(ica, ax_topo, self.mne.picks[:self.mne.n_channels])

        # SAVE UI ELEMENT HANDLES
        vars(self.mne).update(
            ax_main=ax_main,
            ax_help=ax_help,
            ax_proj=ax_proj,
            ax_hscroll=ax_hscroll,
            ax_vscroll=ax_vscroll,
            vsel_patch=vsel_patch,
            hsel_patch=hsel_patch,
            vline=vline,
            vline_hscroll=vline_hscroll,
            vline_text=vline_text,
        )


    def _update_picks(self):
        """Compute which channel indices to show."""
        n_extra_chans = int(np.sum([1 for k, ch_type in enumerate(self.mne.ch_types) if ch_type == 'eog' or ch_type == 'ecg']))
        if self.mne.butterfly and self.mne.ch_selections is not None:
            selections_dict = self._make_butterfly_selections_dict()
            self.mne.picks = np.concatenate(tuple(selections_dict.values()))
        elif self.mne.butterfly:
            self.mne.picks = self.mne.ch_order
        else:
            # this is replaced:
            # _slice = slice(self.mne.picks[n_extra_chans],
            #                self.mne.picks[n_extra_chans] + self.mne.n_channels)
                        # self.mne.picks = self.mne.ch_order[_slice]

            _slice = slice(self.mne.ch_start,
                           self.mne.ch_start + self.mne.n_channels - n_extra_chans )
            self.mne.picks = np.concatenate([np.arange(n_extra_chans), self.mne.ch_order[_slice]])
            self.mne.n_channels = len(self.mne.picks)
        assert isinstance(self.mne.picks, np.ndarray)
        assert self.mne.picks.dtype.kind == 'i'
    
    def _draw_traces(self):
        """Draw (or redraw) the channel data."""
        
        from matplotlib.colors import to_rgba_array
        from matplotlib.patches import Rectangle

        # OSL ADDITION
        from mne import pick_types
        

        # clear scalebars
        if self.mne.scalebars_visible:
            self._hide_scalebars()
        # get info about currently visible channels
        picks = self.mne.picks
        ch_names = self.mne.ch_names[picks]
        ch_types = self.mne.ch_types[picks]
        offset_ixs = (picks
                      if self.mne.butterfly and self.mne.ch_selections is None
                      else slice(None))
        offsets = self.mne.trace_offsets[offset_ixs]
        bad_bool = np.in1d(ch_names, self.mne.info["bads"])
        # OSL ADDITION
        bad_int = list(np.ones(len(picks))*-1)
        extra_chans = [picks[k]  for k, ch_type in enumerate(ch_types) if ch_type == 'eog' or ch_type=='ecg']
        for cnt, ch in enumerate([self.mne.ch_names[ii] for ii in picks]):
            if cnt < len(extra_chans):
                continue
            i = self.mne.ica._ica_names.index(ch)
            if ch in self.mne.info["bads"]:
                if len(list(self.mne.ica.labels_.values())) > 0 and i in np.concatenate(list(self.mne.ica.labels_.values())):
                    i = int(i)
                    ix = np.where([i in self.mne.ica.labels_[k] for k in self.mne.ica.labels_.keys()])[0][0]
                    lbl = list(self.mne.ica.labels_.keys())[ix].split('/')[0]
                    if lbl == 'unknown':
                        bad_int[cnt] = int(0)
                    else:
                        bad_int[cnt] = int(self.mne.bad_labels_list.index(lbl) + 1)
                else:
                    bad_int[cnt] = int(0)
            else:
                if len(list(self.mne.ica.labels_.values())) > 0 and i in np.concatenate(list(self.mne.ica.labels_.values())):  # remove entry
                    i = int(i)
                    whichkeys = [list(self.mne.ica.labels_.keys())[k] for k in np.where([i in self.mne.ica.labels_[k] for k in self.mne.ica.labels_.keys()])[0]]
                    for k in whichkeys:
                        self.mne.ica.labels_[k] = list(np.setdiff1d(self.mne.ica.labels_[k], i))
                bad_int[cnt] = -1

        # colors
        good_ch_colors = [self.mne.ch_color_dict[_type] for _type in ch_types]
        c = [
            self.mne.ch_color_bad
        ] + self.mne.bad_label_colors  # OSL ADDITION: match colors to specific artifact labels
        ch_colors = to_rgba_array(
            [c[_bad] if _bad >= 0 else _color for _bad, _color in zip(bad_int, good_ch_colors)])
        self.mne.ch_colors = np.array(good_ch_colors)  # use for unmarking bads
        labels = self.mne.ax_main.yaxis.get_ticklabels()
        if self.mne.butterfly:
            for label in labels:
                label.set_color(self.mne.fgcolor)
        else:
            for label, color in zip(labels, ch_colors):
                label.set_color(color)
        # decim
        decim = np.ones_like(picks)
        data_picks_mask = np.in1d(picks, self.mne.picks_data)
        decim[data_picks_mask] = self.mne.decim
        # decim can vary by channel type, so compute different `times` vectors
        decim_times = {
            decim_value: self.mne.times[::decim_value] + self.mne.first_time
            for decim_value in set(decim)
        }
        # add more traces if needed
        n_picks = len(picks)
        if n_picks > len(self.mne.traces):
            n_new_chs = n_picks - len(self.mne.traces)
            new_traces = self.mne.ax_main.plot(
                np.full((1, n_new_chs), np.nan), **self.mne.trace_kwargs
            )
            self.mne.traces.extend(new_traces)
        # remove extra traces if needed
        extra_traces = self.mne.traces[n_picks:]
        for trace in extra_traces:
            self.mne.ax_main.lines.remove(trace)
        self.mne.traces = self.mne.traces[:n_picks]

        # check for bad epochs
        time_range = (self.mne.times + self.mne.first_time)[[0, -1]]
        if self.mne.instance_type == "epochs":
            epoch_ix = np.searchsorted(self.mne.boundary_times, time_range)
            epoch_ix = np.arange(epoch_ix[0], epoch_ix[1])
            epoch_nums = self.mne.inst.selection[epoch_ix[0] : epoch_ix[-1] + 1]
            visible_bad_epochs = epoch_nums[
                np.in1d(epoch_nums, self.mne.bad_epochs).nonzero()
            ]
            while len(self.mne.epoch_traces):
                _trace = self.mne.epoch_traces.pop(-1)
                self.mne.ax_main.lines.remove(_trace)
            # handle custom epoch colors (for autoreject integration)
            if self.mne.epoch_colors is None:
                # shape: n_traces × RGBA → n_traces × n_epochs × RGBA
                custom_colors = np.tile(
                    ch_colors[:, None, :], (1, self.mne.n_epochs, 1)
                )
            else:
                custom_colors = np.empty((len(self.mne.picks), self.mne.n_epochs, 4))
                for ii, _epoch_ix in enumerate(epoch_ix):
                    this_colors = self.mne.epoch_colors[_epoch_ix]
                    custom_colors[:, ii] = to_rgba_array(
                        [this_colors[_ch] for _ch in picks]
                    )
            # override custom color on bad epochs
            for _bad in visible_bad_epochs:
                _ix = epoch_nums.tolist().index(_bad)
                _cols = np.array([self.mne.epoch_color_bad, self.mne.ch_color_bad])[
                    bad_bool.astype(int)
                ]
                custom_colors[:, _ix] = to_rgba_array(_cols)

        # update traces
        ylim = self.mne.ax_main.get_ylim()
        for ii, line in enumerate(self.mne.traces):
            this_name = ch_names[ii]
            this_type = ch_types[ii]
            this_offset = self.mne.trace_offsets[ii]
            this_times = decim_times[decim[ii]]
            this_data = this_offset - self.mne.data[ii] * self.mne.scale_factor
            this_data = this_data[..., :: decim[ii]]
            # clip
            if self.mne.clipping == "clamp":
                this_data = np.clip(this_data, -0.5, 0.5)
            elif self.mne.clipping is not None:
                clip = self.mne.clipping * (0.2 if self.mne.butterfly else 1)
                bottom = max(this_offset - clip, ylim[1])
                height = min(2 * clip, ylim[0] - bottom)
                rect = Rectangle(
                    xy=np.array([time_range[0], bottom]),
                    width=time_range[1] - time_range[0],
                    height=height,
                    transform=self.mne.ax_main.transData,
                )
                line.set_clip_path(rect)
            # prep z order
            is_bad_ch = this_name in self.mne.info["bads"]
            this_z = self.mne.zorder["bads" if is_bad_ch else "data"]
            if self.mne.butterfly and not is_bad_ch:
                this_z = self.mne.zorder.get(this_type, this_z)
            # plot each trace multiple times to get the desired epoch coloring.
            # use masked arrays to plot discontinuous epochs that have the same
            # color in a single plot() call.
            if self.mne.instance_type == "epochs":
                this_colors = custom_colors[ii]
                for cix, color in enumerate(np.unique(this_colors, axis=0)):
                    bool_ixs = (this_colors == color).all(axis=1)
                    mask = np.zeros_like(this_times, dtype=bool)
                    _starts = self.mne.boundary_times[epoch_ix][bool_ixs]
                    _stops = self.mne.boundary_times[epoch_ix + 1][bool_ixs]
                    for _start, _stop in zip(_starts, _stops):
                        _mask = np.logical_and(_start < this_times, this_times <= _stop)
                        mask = mask | _mask
                    _times = np.ma.masked_array(this_times, mask=~mask)
                    # always use the existing traces first
                    if cix == 0:
                        line.set_xdata(_times)
                        line.set_ydata(this_data)
                        line.set_color(color)
                        line.set_zorder(this_z)
                    else:  # make new traces as needed
                        _trace = self.mne.ax_main.plot(
                            _times,
                            this_data,
                            color=color,
                            zorder=this_z,
                            **self.mne.trace_kwargs,
                        )
                        self.mne.epoch_traces.extend(_trace)
            else:
                line.set_xdata(this_times)
                line.set_ydata(this_data)
                line.set_color(ch_colors[ii])
                line.set_zorder(this_z)
        # update xlim
        self.mne.ax_main.set_xlim(*time_range)
        # draw scalebars maybe
        if self.mne.scalebars_visible:
            self._show_scalebars()
        # redraw event lines
        if self.mne.event_times is not None:
            self._draw_event_lines()

        # OSL ADDITION: ADD TOPOS:
        exist_meg = any(ct in np.unique(self.mne.ica.get_channel_types()) for ct in ['mag', 'grad'])
        exist_eeg = 'eeg' in np.unique(self.mne.ica.get_channel_types())
        n_topos = len(picks)
        n_chtype = len(
            np.unique(
                [
                    channel_type(self.mne.ica.info, ch)
                    for ch in pick_types(self.mne.ica.info, meg=exist_meg, eeg=exist_eeg)
                ]
            )
        )
        ax_topo = np.reshape(
            self.get_axes()[1 : n_topos * n_chtype + 1], (n_chtype, n_topos)
        )
        self.plot_topos(self.mne.ica, ax_topo, self.mne.picks)
        
        # OSL ADDITION: ADD BAD LABELS
        for i in range(len(self.mne.bad_labels_list)+2):
            if i == 0:
                plt.figtext(self.mne.bad_labels_xpos, self.mne.bad_labels_ypos[i], "bad component \ntype:", fontweight='bold')
            elif i == 1:
                plt.figtext(self.mne.bad_labels_xpos, self.mne.bad_labels_ypos[i], "unknown",
                            color=self.mne.ch_color_bad, fontweight='semibold')
            else:
                plt.figtext(self.mne.bad_labels_xpos, self.mne.bad_labels_ypos[i], f'{i-1}: ' + self.mne.bad_labels_list[i - 2],
                            color=self.mne.bad_label_colors[i - 2], fontweight='semibold')

        self._update_vscroll() # takes care of the vsel_patch, because it's too big when there's extra chans

    def plot_topos(self, ica, ax_topo, picks):  # OSL ADDITION FOR TOPOS
        import mne
        from mne.viz.topomap import _plot_ica_topomap

        extra_chans = [k for k, ch_type in enumerate(self.mne.ch_types[picks]) if ch_type == 'eog' or ch_type == 'ecg']
        exist_meg = any(ct in np.unique(ica.get_channel_types()) for ct in ['mag', 'grad'])
        exist_eeg = 'eeg' in np.unique(ica.get_channel_types())
        n_topos = len(picks)
        ica_tmp = ica.copy()
        ica_tmp._ica_names = ["" for i in ica_tmp._ica_names]
        nchans, ncomps = ica_tmp.get_components().shape
        chtype = np.unique(
            [
                channel_type(ica.info, ch)
                for ch in mne.pick_types(ica.info, meg=exist_meg, eeg=exist_eeg)
            ]
        )
        n_chtype = len(chtype)
        for i in range(n_chtype):
            for j in range(n_topos):
                if picks[j]<len(extra_chans):
                    ax_topo[i, j].clear()
                    ax_topo[i, j].set_axis_off()
                else:
            
                    _plot_ica_topomap(
                        ica_tmp,
                        idx=picks[j]-len(extra_chans),
                        ch_type=chtype[i],
                        axes=ax_topo[i, j],
                        vmin=None,
                        vmax=None,
                        cmap="RdBu_r",
                        colorbar=False,
                        title=None,
                        show=True,
                        outlines="head",
                        contours=0,
                        image_interp="cubic",
                        res=64,
                        sensors=False,
                        allow_ref_meg=False,
                        sphere=None,
                    )
                if j==0:
                        ax_topo[i, j].set_title(f"{chtype[i]}")
                else:
                    ax_topo[i, j].set_title('')

    def _keypress(self, event):
        from mne.viz.utils import _events_off

        """Handle keypress events."""
        key = event.key
        n_channels = self.mne.n_channels
        n_extra_chans = int(np.sum([1 for k, ch_type in enumerate(self.mne.ch_types) if ch_type == 'eog' or ch_type == 'ecg']))
        if self.mne.is_epochs:
            last_time = self.mne.n_times / self.mne.info["sfreq"]
        else:
            last_time = self.mne.inst.times[-1]
        # scroll up/down
        if key in ('down', 'up', 'shift+down', 'shift+up'):
            key = key.split('+')[-1]
            direction = -1 if key == 'up' else 1
            # butterfly case
            if self.mne.butterfly:
                return
            # group_by case
            elif self.mne.fig_selection is not None:
                buttons = self.mne.fig_selection.mne.radio_ax.buttons
                labels = [label.get_text() for label in buttons.labels]
                current_label = buttons.value_selected
                current_idx = labels.index(current_label)
                selections_dict = self.mne.ch_selections
                penult = current_idx < (len(labels) - 1)
                pre_penult = current_idx < (len(labels) - 2)
                has_custom = selections_dict.get('Custom', None) is not None
                def_custom = len(selections_dict.get('Custom', list()))
                up_ok = key == 'up' and current_idx > 0
                down_ok = key == 'down' and (
                        pre_penult or
                        (penult and not has_custom) or
                        (penult and has_custom and def_custom))
                if up_ok or down_ok:
                    buttons.set_active(current_idx + direction)
            # normal case
            else:
                ceiling = len(self.mne.ch_order) - (n_channels - n_extra_chans)
                ch_start = self.mne.ch_start + direction * (n_channels - n_extra_chans)
                self.mne.ch_start = np.clip(ch_start, n_extra_chans, ceiling)
                self._update_picks() 
                self._update_vscroll()
                self._redraw()
        # scroll left/right
        elif key in ("right", "left", "shift+right", "shift+left"):
            old_t_start = self.mne.t_start
            direction = 1 if key.endswith("right") else -1
            if self.mne.is_epochs:
                denom = 1 if key.startswith("shift") else self.mne.n_epochs
            else:
                denom = 1 if key.startswith("shift") else 4
            t_max = last_time - self.mne.duration
            t_start = self.mne.t_start + direction * self.mne.duration / denom
            self.mne.t_start = np.clip(t_start, self.mne.first_time, t_max)
            if self.mne.t_start != old_t_start:
                self._update_hscroll()
                self._redraw(annotations=True)
        # scale traces
        elif key in ("=", "+", "-"):
            scaler = 1 / 1.1 if key == "-" else 1.1
            self.mne.scale_factor *= scaler
            self._redraw(update_data=False)
        # change number of visible channels
        elif (
                key in ("pageup", "pagedown")
                and self.mne.fig_selection is None
                and not self.mne.butterfly
        ):
            new_n_ch = n_channels + (1 if key == "pageup" else -1)
            self.mne.n_channels = np.clip(new_n_ch, 1, len(self.mne.ch_order))
            # add new chs from above if we're at the bottom of the scrollbar
            ch_end = self.mne.ch_start + self.mne.n_channels
            if ch_end > len(self.mne.ch_order) and self.mne.ch_start > 0:
                self.mne.ch_start -= 1
                self._update_vscroll()
            # redraw only if changed
            if self.mne.n_channels != n_channels:
                self._update_picks()
                self._update_trace_offsets()
                self._redraw(annotations=True)
        # change duration
        elif key in ("home", "end"):
            old_dur = self.mne.duration
            dur_delta = 1 if key == "end" else -1
            if self.mne.is_epochs:
                # prevent from showing zero epochs, or more epochs than we have
                self.mne.n_epochs = np.clip(
                    self.mne.n_epochs + dur_delta, 1, len(self.mne.inst)
                )
                # use the length of one epoch as duration change
                min_dur = len(self.mne.inst.times) / self.mne.info["sfreq"]
                new_dur = self.mne.duration + dur_delta * min_dur
            else:
                # never show fewer than 3 samples
                min_dur = 3 * np.diff(self.mne.inst.times[:2])[0]
                # use multiplicative dur_delta
                dur_delta = 5 / 4 if dur_delta > 0 else 4 / 5
                new_dur = self.mne.duration * dur_delta
            self.mne.duration = np.clip(new_dur, min_dur, last_time)
            if self.mne.duration != old_dur:
                if self.mne.t_start + self.mne.duration > last_time:
                    self.mne.t_start = last_time - self.mne.duration
                self._update_hscroll()
                self._redraw(annotations=True)
        elif key == "?":  # help window
            self._toggle_help_fig(event)
        elif key == "a":  # annotation mode
            self._toggle_annotation_fig()
        elif key == "b" and self.mne.instance_type != "ica":  # butterfly mode
            self._toggle_butterfly()
        elif key == "d":  # DC shift
            self.mne.remove_dc = not self.mne.remove_dc
            self._redraw()
        elif key == "h" and self.mne.instance_type == "epochs":  # histogram
            self._toggle_epoch_histogram()
        elif key == "j" and len(self.mne.projs):  # SSP window
            self._toggle_proj_fig()
        elif key == 'J' and len(self.mne.projs):
            self._toggle_proj_checkbox(event, toggle_all=True)
        elif key == "p":  # toggle draggable annotations
            self._toggle_draggable_annotations(event)
            if self.mne.fig_annotation is not None:
                checkbox = self.mne.fig_annotation.mne.drag_checkbox
                with _events_off(checkbox):
                    checkbox.set_active(0)
        elif key == "s":  # scalebars
            self._toggle_scalebars(event)
        elif key == "w":  # toggle noise cov whitening
            self._toggle_whitening()
        elif key == "z":  # zen mode: hide scrollbars and buttons
            self._toggle_scrollbars()
            self._redraw(update_data=False)
        elif key == "t":
            self._toggle_time_format()
        # OSL ADDITION: labeling artifact type of bad components
        elif str(key).isnumeric() and (
                int(key) in range(len(self.mne.bad_labels_list) + 1)
        ):
            if len(self.mne.info["bads"]) > 0 and self.mne.info["bads"][-1] in self.mne.ica._ica_names:
                last_bad_component = self.mne.ica._ica_names.index(self.mne.info["bads"][-1])
                all_labels = list(self.mne.ica.labels_.keys())

                # first remove from a the key it was in before, if applicable:
                if len(list(self.mne.ica.labels_.values())) > 0 and last_bad_component in list(self.mne.ica.labels_.values())[0]:
                    ix = \
                    np.where([last_bad_component in self.mne.ica.labels_[k] for k in self.mne.ica.labels_.keys()])[0]
                    for ixx in ix:
                        lbl = all_labels[ix]
                        self.mne.ica.labels_[lbl] = np.setdiff1d(self.mne.ica.labels_[lbl], last_bad_component)

                # create label based on label list and put it into MNE style
                tmp_label = self.mne.bad_labels_list[
                    int(key) - 1]
                if tmp_label == 'eog':
                    tmp_label = tmp_label + '/3/manual'
                else:
                    tmp_label = tmp_label + '/manual'

                # save bad component label in corresponding dict.
                if tmp_label in self.mne.ica.labels_ and len(self.mne.ica.labels_[tmp_label]) > 0:
                    self.mne.ica.labels_[tmp_label].append(last_bad_component)
                else:
                    self.mne.ica.labels_[tmp_label] = [last_bad_component]
            self._draw_traces()  # This makes sure the traces are given the corresponding color right away
        else:  # check for close key / fullscreen toggle
            super()._keypress(event)

    def _update_vscroll(self):
        """Update the vertical scrollbar (channel) selection indicator."""
        n_extra_chans = int(np.sum([1 for k, ch_type in enumerate(self.mne.ch_types) if ch_type == 'eog' or ch_type == 'ecg']))
        self.mne.vsel_patch.set_xy((0, self.mne.ch_start - n_extra_chans))
        self.mne.vsel_patch.set_height(self.mne.n_channels - n_extra_chans)
        self._update_yaxis_labels()
    
    def _close(self, event):
        # OSL VERSION - SIMILAR TO OLD MNE VERSION TODO: Check if we need to adopt this
        """Handle close events (via keypress or window [x])."""
        from matplotlib.pyplot import close
        from mne.utils import set_config

        # write out bad epochs (after converting epoch numbers to indices)
        if self.mne.instance_type == "epochs":
            bad_ixs = np.in1d(self.mne.inst.selection, self.mne.bad_epochs).nonzero()[0]
            self.mne.inst.drop(bad_ixs)
        # write bad channels back to instance (don't do this for proj;
        # proj checkboxes are for viz only and shouldn't modify the instance)
        if self.mne.instance_type in ("raw", "epochs"):
            self.mne.inst.info["bads"] = self.mne.info["bads"]

        # OSL ADDITION
        # ICA excludes
        elif self.mne.instance_type == "ica":
            # remove artefact channels from exclude (if present)
            rm = []
            for cnt, ch in enumerate(self.mne.info['bads']):
                if ch not in self.mne.ica._ica_names:
                    rm.append(cnt)
            [self.mne.info['bads'].pop(i) for i in np.sort(rm)[::-1]]
            self.mne.ica.exclude = [
                self.mne.ica._ica_names.index(ch) for ch in self.mne.info["bads"]
            ]
            # OSL ADDITION: remove bad component labels that were reversed to good component
            tmp = list(self.mne.ica.labels_.values())[:]
            try:
                tmp = np.unique(np.concatenate(tmp))
            except:
                tmp = []

            for ch in tmp:
                ch = int(ch)
                if ch not in self.mne.ica.exclude:
                    # find in which label it has
                    allix = np.where([ch in self.mne.ica.labels_[key] for key in self.mne.ica.labels_.keys()])[0]
                    for ix in allix:
                        self.mne.ica.labels_[list(self.mne.ica.labels_.keys())[ix]] = \
                            np.setdiff1d(self.mne.ica.labels_[list(self.mne.ica.labels_.keys())[ix]], ch)
            
            # label bad components without a manual label as "unknown"
            for ch in self.mne.ica.exclude:
                ch = int(ch)
                tmp = list(self.mne.ica.labels_.values())
                if len(tmp)==0:
                    tmp = []
                else:
                    tmp = np.concatenate(tmp)
                if ch not in tmp:
                    if "unknown" not in self.mne.ica.labels_.keys():
                        self.mne.ica.labels_["unknown"] = []
                    self.mne.ica.labels_["unknown"] = list(self.mne.ica.labels_["unknown"])
                    self.mne.ica.labels_["unknown"].append(ch)
                    
              
            # Add to labels_ a generic eog/ecg field
            if len(list(self.mne.ica.labels_.keys())) > 0:
                if "ecg" not in self.mne.ica.labels_:
                    self.mne.ica.labels_["ecg"] = []
                if "eog" not in self.mne.ica.labels_:
                    self.mne.ica.labels_["eog"] = []
                for key in self.mne.ica.labels_.keys():
                    self.mne.ica.labels_[key] = list(self.mne.ica.labels_[key])
                
                for key in self.mne.ica.labels_.keys():
                    self.mne.ica.labels_[key] = list(self.mne.ica.labels_[key])
                
                for k in list(self.mne.ica.labels_.keys()):
                    if "ecg" in k.lower() and k.lower() != "ecg":
                        tmp = self.mne.ica.labels_[k]
                        if type(tmp) is list and tmp:
                            tmp = tmp[0]
                        self.mne.ica.labels_["ecg"].append(tmp)
                    elif "eog" in k.lower() and k.lower() != "eog":
                        tmp = self.mne.ica.labels_[k]
                        if type(tmp) is list and tmp:
                            tmp = tmp[0]
                        self.mne.ica.labels_["eog"].append(tmp)       
                self.mne.ica.labels_["ecg"] = [v for v in self.mne.ica.labels_["ecg"] if v!= []]
                self.mne.ica.labels_["eog"] = [v for v in self.mne.ica.labels_["eog"] if v!= []]
                self.mne.ica.labels_["ecg"] = np.unique(self.mne.ica.labels_["ecg"]).tolist()
                self.mne.ica.labels_["eog"] = np.unique(self.mne.ica.labels_["eog"]).tolist()
                for key in self.mne.ica.labels_.keys():
                    self.mne.ica.labels_[key] = list(self.mne.ica.labels_[key])
                    
        # write logs
        logger.info(f"Components marked as bad: {sorted(self.mne.ica.exclude) or 'none'}")
        for lb in self.mne.ica.labels_.keys():
            if 'manual' in lb or lb=='unknown':
                logger.info(f"Components manually labeled as '{lb.split('/')[0]}': {sorted(self.mne.ica.labels_[lb])}")    
        
        # write window size to config
        size = ",".join(self.get_size_inches().astype(str))
        set_config("MNE_BROWSE_RAW_SIZE", size, set_env=False)
        # Clean up child figures (don't pop(), child figs remove themselves)
        while len(self.mne.child_figs):
            fig = self.mne.child_figs[-1]
            close(fig)

def flatten_recursive(lst):
    """Flatten a list using recursion."""
    for item in lst:
        if isinstance(item, list):
            yield from flatten_recursive(item)
        else:
            yield item


# TODO: OSL IMPLEMENT PLOT_ICA FOR EVOKED DATA
def _plot_ica_sources_evoked(
    evoked,
    picks,
    exclude,
    title,
    show,
    ica,
    labels=None,
    n_channels=10,
    bad_labels_list=None,
):
    """Plot average over epochs in ICA space.

    Parameters
    ----------
    evoked : instance of mne.Evoked
        The Evoked to be used.
    %(picks_base)s all sources in the order as fitted.
    exclude : array-like of int
        The components marked for exclusion. If None (default), ICA.exclude
        will be used.
    title : str
        The figure title.
    show : bool
        Show figure if True.
    labels : None | dict
        The ICA labels attribute.
    """
    raise ValueError("plot_ica is not yet supported for Evoked data")

    import matplotlib.pyplot as plt
    from matplotlib import patheffects

    if title is None:
        title = "Reconstructed latent sources, time-locked"

    fig, axes = plt.subplots(1)
    ax = axes
    axes = [axes]
    times = evoked.times * 1e3

    # plot unclassified sources and label excluded ones
    lines = list()
    texts = list()
    picks = np.sort(picks)
    idxs = [picks]

    if labels is not None:
        labels_used = [k for k in labels if "/" not in k]

    exclude_labels = list()
    for ii in picks:
        if ii in exclude:
            line_label = ica._ica_names[ii]
            if labels is not None:
                annot = list()
                for this_label in labels_used:
                    indices = labels[this_label]
                    if ii in indices:
                        annot.append(this_label)

                if annot:
                    line_label += " – " + ", ".join(annot)  # Unicode en-dash
            exclude_labels.append(line_label)
        else:
            exclude_labels.append(None)
    label_props = [("k", "-") if lb is None else ("r", "-") for lb in exclude_labels]
    styles = ["-", "--", ":", "-."]
    if labels is not None:
        # differentiate categories by linestyle and components by color
        col_lbs = [it for it in exclude_labels if it is not None]
        cmap = plt.get_cmap("tab10", len(col_lbs))

        unique_labels = set()
        for label in exclude_labels:
            if label is None:
                continue
            elif " – " in label:
                unique_labels.add(label.split(" – ")[1])
            else:
                unique_labels.add("")

        # Determine up to 4 different styles for n categories
        cat_styles = dict(
            zip(
                unique_labels,
                map(
                    lambda ux: styles[int(ux % len(styles))], range(len(unique_labels))
                ),
            )
        )
        for label_idx, label in enumerate(exclude_labels):
            if label is not None:
                color = cmap(col_lbs.index(label))
                if " – " in label:
                    label_name = label.split(" – ")[1]
                else:
                    label_name = ""
                style = cat_styles[label_name]
                label_props[label_idx] = (color, style)

    for exc_label, ii in zip(exclude_labels, picks):
        color, style = label_props[ii]
        # ensure traces of excluded components are plotted on top
        zorder = 2 if exc_label is None else 10
        lines.extend(
            ax.plot(
                times,
                evoked.data[ii].T,
                picker=True,
                zorder=zorder,
                color=color,
                linestyle=style,
                label=exc_label,
            )
        )
        lines[-1].set_pickradius(3.0)

    ax.set(title=title, xlim=times[[0, -1]], xlabel="Time (ms)", ylabel="(NA)")
    if len(exclude) > 0:
        plt.legend(loc="best")
    tight_layout(fig=fig)

    texts.append(
        ax.text(
            0,
            0,
            "",
            zorder=3,
            verticalalignment="baseline",
            horizontalalignment="left",
            fontweight="bold",
            alpha=0,
        )
    )
    # this is done to give the structure of a list of lists of a group of lines
    # in each subplot
    lines = [lines]
    ch_names = evoked.ch_names

    path_effects = [patheffects.withStroke(linewidth=2, foreground="w", alpha=0.75)]
    params = dict(
        axes=axes,
        texts=texts,
        lines=lines,
        idxs=idxs,
        ch_names=ch_names,
        need_draw=False,
        path_effects=path_effects,
    )
    fig.canvas.mpl_connect("pick_event", partial(_butterfly_onpick, params=params))
    fig.canvas.mpl_connect(
        "button_press_event", partial(_butterfly_on_button_press, params=params)
    )
    plt_show(show)
    return fig
