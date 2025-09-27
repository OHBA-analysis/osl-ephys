import mne
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.sparse import csr_array
from itertools import compress
from copy import deepcopy

from matplotlib.colors import ListedColormap
from matplotlib.patches import ConnectionPatch
from nilearn.plotting import plot_glass_brain, plot_markers

import glmtools as glm
import nibabel as nib

from ..source_recon import parcellation


class GLMBaseResult:
    """A class for GLM-Epochs fitted to MNE-Python Raw objects."""
    def __init__(self, model, design, info, data=None):
        """
        Parameters
        ----------
        model : :py:class:`glmtools.fit.OLSModel <glmtools.fit.OLSModel>'
            The  model object.
        design : :py:class:`glmtools.design.GLMDesign <glmtools.design.GLMDesign>`
            The  GLM design object.
        info : :py:class:`mne.Info <mne.Info>`
            The MNE-Python Info object.
            
            
        References
        ----------
        https://gitlab.com/ajquinn/glmtools/-/tree/master
        
        """
        self.model = model
        self.design = design
        self.data = data
        self.info = info


    def compute_cohens_f2(self, reg_ind):
        """Compute a Cohens F-squared effect size associated with a specific regressor.


        Parameters
        ----------
        reg_ind : int
            regressor of interest

        Returns
        -------
        array
            Cohen's F-squared values

        """
        return glm.fit.cohens_f2(self.design, self.data, reg_ind, model=self.model)


    def save_pkl(self, outname, overwrite=True, save_data=False):
        """Save GLM-Epochs result to a pickle file.

        Parameters
        ----------
        outname : str
             Filename or full file path to write pickle to
        overwrite : bool
             Overwrite previous file if one exists? (Default value = True)
        save_data : bool
             Save epoch data in pickle? This is omitted by default to save disk
             space (Default value = False)

        """
        if Path(outname).exists() and not overwrite:
            msg = "{} already exists. Please delete or do use overwrite=True."
            raise ValueError(msg.format(outname))

        if hasattr(self, 'config'):
            self.config.detrend_func = None  # Have to drop this to pickle glm-spectra

        # This is hacky - but pickles are all or nothing and I don't know how
        # else to do it. HDF5 would be better longer term
        if save_data == False:
            # Temporarily remove data before saving
            dd = self.data
            self.data = None

        with open(outname, 'bw') as outp:
            pickle.dump(self, outp)

        # Put data back
        if save_data == False:
            self.data = dd


class GroupGLMBaseResult:
    """A class for group level GLM-Epochs fitted across mmultiple first-level
    GLM-Epochs computed from MNE-Python Raw objects"""

    def __init__(self, model, design, info, config, fl_contrast_names=None, data=None):
        """
        Parameters
        ----------
        model : :py:class:`glmtools.fit.OLSModel <glmtools.fit.OLSModel>'
            The  model object.
        design : :py:class:`glmtools.design.GLMDesign <glmtools.design.GLMDesign>`
            The  GLM design object.
        info : :py:class:`mne.Info <mne.Info>`
            The MNE-Python Info object.
        config : :py:class:`glmtools.config.GLMConfig <glmtools.config.GLMConfig>`
            The  GLM configuration object.
        fl_contrast_names : list of str
            The names of the first-level contrasts.
        data : :py:class:`mne.Epochs <mne.Epochs>` or :py:class:`mne.Evoked <mne.Evoked>`
            The MNE-Python Epochs or Evoked object.
            
        """
        
        self.model = model
        self.design = design
        self.data = data
        self.config = config

        self.info = info

        # A proper group-model in glmtools will simplify this
        self.contrast_names = self.model.contrast_names
        if fl_contrast_names is None:
            self.fl_contrast_names = [chr(65 + ii) for ii in range(self.model.copes.shape[1])]
        else:
            self.fl_contrast_names = fl_contrast_names

    def get_channel_adjacency(self, dist=40):
        """Return adjacency matrix of channels.
        
        Parameters
        ----------
        dist : float
            Distance in mm between parcel centroids to consider neighbours.
            Only used if data is parcellated.
        
        Returns
        -------
        adjacency : scipy.sparse.csr_matrix
            The adjacency matrix.
        ch_names : list of str
            The channel names.
        """
        if np.any(['parcel' in ch for ch in self.info['ch_names']]):
            # We have parcellated data
            parcellation_file = parcellation.guess_parcellation(int(np.sum(['parcel' in ch for ch in self.info['ch_names']])))
            adjacency = csr_array(parcellation.spatial_dist_adjacency(parcellation_file, dist=dist))
        elif np.any(['state' in ch for ch in self.info['ch_names']]) or np.any(['mode' in ch for ch in self.info['ch_names']]):
            adjacency = csr_array(np.eye(len(self.info['ch_names'])))
        else:
            channel_types = [mne.channel_type(self.info, idx) for idx in range(self.info['nchan'])]
            ch_type = channel_types[0]  # Assuming these are all the same!
            adjacency, ch_names = mne.channels.channels._compute_ch_adjacency(self.info, ch_type)
        ntests = np.prod(self.data.data.shape[2:])
        ntimes = self.data.data.shape[3]
        print('{} : {}'.format(ntimes, ntests))
        return mne.stats.cluster_level._setup_adjacency(adjacency, ntests, ntimes)


class BaseSensorPerm:
    """A base class for sensor x frequency and sensor x time permutation tests computed from a
    group level GLM-Spectrum."""
    def save_pkl(self, outname, overwrite=True, save_data=False):
        """Save GLM-Epochs result to a pickle file.

        Parameters
        ----------
        outname : str
                Filename or full file path to write pickle to
        overwrite : bool
                Overwrite previous file if one exists? (Default value = True)
        save_data : bool
                Save epoch data in pickle? This is omitted by default to save disk
                space (Default value = False)

        """
        if Path(outname).exists() and not overwrite:
            msg = "{} already exists. Please delete or do use overwrite=True."
            raise ValueError(msg.format(outname))

        with open(outname, 'bw') as outp:
            pickle.dump(self, outp)


class SensorMaxStatPerm(BaseSensorPerm):
    """A class holding the result for sensor x frequency or sensor x time max-stat permutation test computed
    from a group level GLM-Spectrum or GLM-Epochs"""

    def __init__(self, glmsp, gl_con, fl_con=0, nperms=1000,
                    tstat_args=None,  metric='tstats', nprocesses=1,
                    pooled_dims=(1,2), tmin=None, tmax=None, fmin=None,
                    fmax=None, picks=None):
        """Initialise the SensorMaxStatPerm class.
        
        Parameters
        ----------
        glmsp : :py:class:`GroupGLMBaseResult <GroupGLMBaseResult>`
            The group level GLM result object.
        gl_con : int
            The index of the group level contrast to test.
        fl_con : int
            The index of the first-level contrast to test.
        nperms : int
            The number of permutations to use.
        tstat_args : dict
            The arguments to pass to the tstat function.
        metric : str
            The metric to use for the permutation test.
        nprocesses : int
            The number of processes to use.
        pooled_dims : tuple of int
            The dimensions to pool over.
        tmin : float or None (default)
            The minimum time to consider.
        tmax : float or None (default)
            The maximum time to consider.
        fmin : float or None (default)
            The minimum frequency to consider.
        fmax : float or None (default)
        picks : list of int or None (default)
            The channel names to consider.
        """
        # There is a major pain here in that MNE stores raw data in [channels x time]
        # but builds adjacencies in [time x channels] - we don't need adjacencies for perms
        # but we do for making clusters for plotting, so here we are
        self.adjacency = glmsp.get_channel_adjacency()
        self.perm_data = glmsp.get_fl_contrast(fl_con)
        self.perm_data.data = np.swapaxes(self.perm_data.data, 1, 2)
        
        # change the dim_labels to match the data (swap time and channels, and remove firstlevel contrast)
        self.perm_data.info['dim_labels'] = [self.perm_data.info['dim_labels'][i] for i in [0,3,2]]
        
        if hasattr(glmsp, 'times'):
            self.times=glmsp.times
            
            # select times
            if tmin is not None:
                self.tmin = tmin
            else:
                self.tmin=self.times[0]
            if tmax is not None:
                self.tmax = tmax
            else:
                self.tmax=self.times[-1]
                
        if hasattr(glmsp, 'f'):
            self.f=glmsp.f
            
            # select frequencies
            if fmin is not None:
                self.fmin = fmin
            else:
                self.fmin=self.f[0]
            if fmax is not None:
                self.fmax = fmax
            else:
                self.fmax=self.f[-1]
                
        # select channels
        if picks is not None:
            self.picks = mne.pick_channels(glmsp.info['ch_names'], picks).astype(int)
        else:
            self.picks = np.arange(len(glmsp.info['ch_names']))
        
        # make selection of the data if needed 
        perm_data = deepcopy(self.perm_data)
        if hasattr(self, 'times'):
            perm_data.data = perm_data.data[:, np.logical_and(self.times>=self.tmin, self.times<=self.tmax), :]
        if hasattr(self, 'f'):
            perm_data.data = perm_data.data[:, np.logical_and(self.f>=self.fmin, self.f<=self.fmax), :]
        perm_data.data = perm_data.data[:, :, self.picks]
        
        self.gl_con = gl_con
        self.fl_con = fl_con
        self.gl_contrast_name = glmsp.contrast_names[gl_con]
        self.fl_contrast_name = glmsp.fl_contrast_names[fl_con]
        self.info = glmsp.info

        self.perms = glm.permutations.MaxStatPermutation(glmsp.design, perm_data, gl_con, nperms,
                                                        nprocesses=nprocesses,
                                                        metric=metric,
                                                        pooled_dims=pooled_dims,
                                                        tstat_args=tstat_args)

    def get_sig_clusters(self, thresh):
        """Return the significant clusters at a given threshold.
        
        Parameters
        ----------
        thresh : float
            The threshold to consider a cluster significant eg 95 or 99
            
        Returns
        -------
        clusters
            A list containing the significant clusters. Each list item contains
            a tuple of three items - the cluster statistic, the cluster
            percentile relative to the null and the spatial/spectral indices of
            the cluster.
        obs_stat
            The observed statistic.
        """
        obs = glm.fit.OLSModel(self.perms._design, self.perm_data)
        obs = obs.get_tstats(**self.perms.tstat_args)[self.gl_con, :, :]
        thresh = self.perms.get_thresh(thresh)

        # select data the permutation test was run on
        obs_sel = deepcopy(obs)
        obs_sel[np.logical_or(self.times<self.tmin, self.times>self.tmax),:] = 0
        obs_sel[:, np.setdiff1d(np.arange(len(self.info.ch_names)), self.picks)] = 0
        
        obs_up = obs_sel.flatten() > thresh
        obs_down = obs_sel.flatten() < -thresh

        from mne.stats.cluster_level import _find_clusters as mne_find_clusters
        from mne.stats.cluster_level import _reshape_clusters as mne_reshape_clusters

        clus_up, cstat_up = mne_find_clusters(obs_up, 0.5, adjacency=self.adjacency)
        clus_up = mne_reshape_clusters(clus_up, obs.shape)

        clus_down, cstat_down = mne_find_clusters(obs_down, 0.5, adjacency=self.adjacency)
        clus_down = mne_reshape_clusters(clus_down, obs.shape)

        # cstat, pval, clu - match cluster stat output
        clusters = []
        for ii in range(len(cstat_down)):
            clusters.append([cstat_down[ii], 0, clus_down[ii]])
        for ii in range(len(cstat_up)):
            clusters.append([cstat_up[ii], 0, clus_up[ii]])

        return clusters, obs
    
    
    def plot_sig_clusters(self, thresh, ax=None, min_extent=1):
        """Plot the significant clusters at a given threshold.

        Parameters
        ----------
        thresh : float
            The threshold to consider a cluster significant eg 95 or 99
        ax : :py:class:`matplotlib.axes <matplotlib.axes>`
            Matplotlib axes to plot on. (Default value = None)
        """
        title = 'group-con: {}\nfirst-level-con: {}'
        title = title.format(self.gl_contrast_name, self.fl_contrast_name)

        clu, obs = self.get_sig_clusters(thresh)
        to_plot = []
        for c in clu:
            to_plot.append(False if len(c[2][0]) < min_extent or len(c[2][1]) < min_extent else True)

        clu = list(compress(clu, to_plot))

        plot_joint_clusters(self.times, obs, clu, self.info, ax=ax, title=title, ylabel='t-stat')



class SensorClusterPerm(BaseSensorPerm):
    """A class holding the result for sensor x frequency or sensor x time cluster stats computed
    from a group level GLM-Spectrum or GLM-Epochs"""

    def __init__(self, glmsp, gl_con, fl_con=0, nperms=1000,
                    cluster_forming_threshold=3, tstat_args=None,
                    metric='tstats', tmin=None, tmax=None, 
                    fmin=None, fmax=None, picks=None, nprocesses=1):
        """Initialise the SensorClusterPerm class.
        
        Parameters
        ----------
        glmsp : :py:class:`GroupGLMBaseResult <GroupGLMBaseResult>`
            The group level GLM result object.
        gl_con : int
            The index of the group level contrast to test.
        fl_con : int
            The index of the first-level contrast to test.
        nperms : int
            The number of permutations to use.
        cluster_forming_threshold : float   
            The threshold to use for cluster forming.
        tstat_args : dict   
            The arguments to pass to the tstat function.
        metric : str
            The metric to use for the permutation test.
        tmin: float or None (default)
            The minimum time to consider.
        tmax: float or None (default)
            The maximum time to consider.
        fmin: float or None (default)
            The minimum frequency to consider.
        fmax: float or None (default)   
            The maximum frequency to consider.
        picks: list of int or None (default)
            The channel names to consider.
        nprocesses : int    
            The number of processes to use.
        
        """

        # There is a major pain here in that MNE stores raw data in [channels x time]
        # but builds adjacencies in [time x channels]
        self.perm_data = glmsp.get_fl_contrast(fl_con)
        self.perm_data.data = np.swapaxes(self.perm_data.data, 1, 2)

        self.gl_contrast_name = glmsp.contrast_names[gl_con]
        self.fl_contrast_name = glmsp.fl_contrast_names[fl_con]
        self.info = glmsp.info
        
        if hasattr(glmsp, 'times'):
            self.times=glmsp.times
            
            # select times
            if tmin is not None:
                self.tmin = tmin
            else:
                self.tmin=self.times[0]
            if tmax is not None:
                self.tmax = tmax
            else:
                self.tmax=self.times[-1]
                
        if hasattr(glmsp, 'f'):
            self.f=glmsp.f
            
            # select frequencies
            if fmin is not None:
                self.fmin = fmin
            else:
                self.fmin=self.f[0]
            if fmax is not None:
                self.fmax = fmax
            else:
                self.fmax=self.f[-1]
                
        # select channels
        if picks is not None:
            self.picks = mne.pick_channels(glmsp.info['ch_names'], picks).astype(int)
        else:
            self.picks = np.arange(len(glmsp.info['ch_names']))
        
        # make selection of the data if needed 
        perm_data = deepcopy(self.perm_data)
        if hasattr(self, 'times'):
            perm_data.data = perm_data.data[:, np.logical_and(self.times>=self.tmin, self.times<=self.tmax), :]
        if hasattr(self, 'f'):
            perm_data.data = perm_data.data[:, np.logical_and(self.f>=self.fmin, self.f<=self.fmax), :]
        perm_data.data = perm_data.data[:, :, self.picks]
        

        adjacency = [[x for x in y if x in self.picks] for y in np.array(glmsp.get_channel_adjacency())[self.picks]]

        self.perms = glm.permutations.MNEClusterPermutation(glmsp.design, perm_data, gl_con, nperms,
                                                        nprocesses=nprocesses,
                                                        metric=metric,
                                                        cluster_forming_threshold=cluster_forming_threshold,
                                                        tstat_args=tstat_args,
                                                        adjacency=adjacency)

    def get_sig_clusters(self, thresh):
        """Return the significant clusters at a given threshold.

        Parameters
        ----------
        thresh : float
            The threshold to consider a cluster significant eg 95 or 99

        Returns
        -------
        clusters
            A list containing the significant clusters. Each list item contains
            a tuple of three items - the cluster statistic, the cluster
            percentile relative to the null and the spatial/spectral indices of
            the cluster.
        obs_stat
            The observed statistic.
        """        
        perm_data = deepcopy(self.perm_data)
        if hasattr(self, 'times'):
            perm_data.data = perm_data.data[:, np.logical_and(self.times>=self.tmin, self.times<=self.tmax), :]
        if hasattr(self, 'f'):
            perm_data.data = perm_data.data[:, np.logical_and(self.f>=self.fmin, self.f<=self.fmax), :]
        perm_data.data = perm_data.data[:, :, self.picks]

        clusters, obs_stat =  self.perms.get_sig_clusters(thresh, perm_data)
        return clusters, obs_stat
    
    
    def plot_sig_clusters(self, thresh, ax=None, min_extent=1):
        """Plot the significant clusters at a given threshold.

        Parameters
        ----------
        thresh : float
            The threshold to consider a cluster significant eg 95 or 99
        ax : :py:class:`matplotlib.axes <matplotlib.axes>`
            Matplotlib axes to plot on. (Default value = None)
        """
        title = 'group-con: {}\nfirst-level-con: {}'
        title = title.format(self.gl_contrast_name, self.fl_contrast_name)

        clu, obs_sel = self.get_sig_clusters(thresh) # obs here is the selected data. We want to plot the full data
        obs = glm.fit.OLSModel(self.perms._design, self.perm_data)
        obs = obs.get_tstats(**self.perms.tstat_args)[self.gl_con, :, :]
        
        to_plot = []
        for c in clu:
            to_plot.append(False if len(c[2][0]) < min_extent or len(c[2][1]) < min_extent else True)

        clu = list(compress(clu, to_plot))

        plot_joint_clusters(self.times, obs, clu, self.info, ax=ax, title=title, ylabel='t-stat')


def plot_joint_clusters(xvect, erp, clusters, info, ax=None, times='auto', 
                                 topo_scale='joint', lw=0.5, ylabel='Power', title='', ylim=None,
                                 xtick_skip=1, topo_prop=1/5, topo_cmap=None, topomap_args=None):
    """Plot a GLM-Epochs contrast from cluster objects, with spatial line colouring and topograpies.

    Parameters
    ----------
    xvect : array_like
        Time vector
    erp : array_like
        epochs values
    clusters : list
        List of cluster objects
    info : dict 
        MNE-Python info object
    ax : {None or axis handle}
        Axis to plot into (Default value = None)
    times : {list, tuple or 'auto'}
        Which times to plot topos for (Default value = 'auto')
    topo_scale : {'joint' or None}  
        Whether to fix topomap colour scales across all topos ('joint') or
        leave them individual (Default value = 'joint')
    lw : float
        Line width(Default value = 0.5)
    ylabel : str
        Y-axis label(Default value = 'Power')
    title : str
        Plot title(Default value = None) 
    ylim : {tuple or list}
        min and max values for y-axis (Default value = None)
    xtick_skip : int
        Number of xaxis ticks to skip, useful for tight plots (Default value = 1)
    topo_prop : float
        Proportion of plot dedicted to topomaps(Default value = 1/3)
    topo_cmap: {None or matplotlib colormap}
        Colormap to use for plotting (Default is 'RdBu_r' if pooled topo data range 
        is positive and negative, otherwise 'Reds' or 'Blues' depending on sign of
        pooled data range)
    """
    if ax is None:
        fig = plt.figure()
        fig.subplots_adjust(top=0.8)
        ax = plt.subplot(111)

    ax.set_axis_off()

    topomap_args = {} if topomap_args is None else topomap_args

    title_prop = 0.1
    main_prop = 1-title_prop-topo_prop
    main_ax = ax.inset_axes((0, 0, 1, main_prop))
    
    plot_sensor_erp(xvect, erp, info, ax=main_ax, lw=0.25, ylabel=ylabel)

    yl = main_ax.get_ylim() if ylim is None else ylim
    yfactor = 1.2 if yl[1] > 0 else 0.8
    main_ax.set_ylim(yl[0], yfactor*yl[1])

    yt = ax.get_yticks()
    inds = yt < yl[1]
    ax.set_yticks(yt[inds])

    ax.figure.canvas.draw()
    offset = ax.yaxis.get_major_formatter().get_offset()
    ax.yaxis.offsetText.set_visible(False)
    ax.text(0, yl[1], offset, ha='right')

    if len(clusters) == 0:
        # put up an empty axes anyway
        topo_pos = [0.3, 1.2, 0.4, 0.4]
        topo = ax.inset_axes(topo_pos, frame_on=False)
        topo.set_xticks([])
        topo.set_yticks([])

    # Reorder clusters in ascending time
    clu_order = []
    for clu in clusters:
        clu_order.append(clu[2][0].min())
    clusters = [clusters[ii] for ii in np.argsort(clu_order)]

    print('\n')
    table_header = '{0:12s}{1:16s}{2:12s}{3:12s}{4:14s}'
    print(table_header.format('Cluster', 'Statistic', 'Time Min', 'Time Max', 'Num Channels'))
    table_template = '{0:<12d}{1:<16.3f}{2:<12.2f}{3:<12.2f}{4:<14d}'

    if type(times)==str and times=='auto':
        topo_centres = np.linspace(0, 1, len(clusters)+2)[1:-1]
        times_topo = times
    else:
        topo_centres = np.linspace(0, 1, len(times)+2)[1:-1]
        times_topo = list(np.sort(times.copy()))
    topo_width = 0.4
    topos = []
    ymax_span = (np.abs(yl[0]) + yl[1]) / (np.abs(yl[0]) + yl[1]*1.2)
    
    data_toplot = []
    topo_ax_toplot = []
    for idx in range(len(clusters)):
        clu = clusters[idx]

        # Extract cluster location in space and time
        channels = np.zeros((erp.shape[1], ))
        channels[clu[2][1]] = 1
        if len(channels) == 204:
            channels = np.logical_or(channels[::2], channels[1::2])
        times = np.zeros((erp.shape[0], ))
        times[clu[2][0]] = 1
        tinds = np.where(times)[0]
        if len(tinds) == 1:
            if tinds[0]<len(xvect)-1: 
                tinds = np.array([tinds[0], tinds[0]+1])
            else: # can't extend to next time point if last time point
                tinds = np.array([tinds[0], tinds[0]])

        msg = 'Cluster {} - stat: {}, time range: {}, num channels {}'
        time_range = (xvect[tinds[0]], xvect[tinds[-1]])
        print(table_template.format(idx+1, clu[0], time_range[0], time_range[1], int(channels.sum())))

        # Plot cluster span overlay on spectrum
        main_ax.axvspan(xvect[tinds[0]], xvect[tinds[-1]], facecolor=[0.7, 0.7, 0.7], alpha=0.5, ymax=ymax_span)
        if type(times_topo)==str and times_topo=='auto':
            tmid = int(np.floor(tinds.mean()))
            # Create topomap axis
            topo_pos = [topo_centres[idx] - 0.2, 1-title_prop-topo_prop, 0.4, topo_prop]
        else:
            tmid_tmp = np.where([itim<=xvect[tinds[-1]] and itim>xvect[tinds[0]] for itim in times_topo])[0]
            if len(tmid_tmp)>0:
                tmid = np.argmin(np.abs(xvect - times_topo[tmid_tmp[0]]))
                # Create topomap axis
                topo_pos = [topo_centres[len(topo_centres)-len(times_topo)] - 0.2, 1-title_prop-topo_prop, 0.4, topo_prop]
                
                times_topo.pop(tmid_tmp[0])
            else:
                continue
        
        # Create topomap axis
        topo_ax = ax.inset_axes(topo_pos)

        # Plot connecting line to topo
        xy_main = (xvect[tmid], yl[1])
        xy_topo = (0.5, 0)
        con = ConnectionPatch(
            xyA=xy_main, coordsA=main_ax.transData,
            xyB=xy_topo, coordsB=topo_ax.transAxes,
            arrowstyle="-", color=[0.7, 0.7, 0.7])
        main_ax.figure.add_artist(con)

        # save the topo axis and data for later
        data_toplot.append(erp[tmid, :])
        topo_ax_toplot.append(topo_ax)
        
    if topo_scale == 'joint' and len(data_toplot) > 0:
        vmin = np.min([t.min() for t in data_toplot])
        vmax = np.max([t.max() for t in data_toplot])
        
        # determine colorbar
        if topo_cmap is None:
            if vmin < 0 and vmax > 0:
                topo_cmap = 'RdBu_r'
                vmin, vmax = np.array([-1,1]) * np.max(np.abs([vmin,vmax]))
            elif vmin >= 0:
                topo_cmap = 'Reds'
            else:
                topo_cmap = 'Blues_r'

        for topo, topo_ax in zip(data_toplot, topo_ax_toplot):
            if np.any(['parcel' in ch for ch in info['ch_names']]): # source level data
                im = plot_source_topo(topo, axis=topo_ax, cmap=topo_cmap) 
            else:
                im, cn = mne.viz.plot_topomap(topo, info, axes=topo_ax, show=False, mask=channels, ch_type='planar1', cmap=topo_cmap)
            im.set_clim(vmin, vmax)
            topos.append(im)

        cb_pos = [0.95, 1-title_prop-topo_prop, 0.025, topo_prop]
        cax =  ax.inset_axes(cb_pos)
        plt.colorbar(topos[0], cax=cax)

    print('\n')  # End table

    ax.set_title(title, x=0.5, y=1-title_prop)
    
    
def plot_sensor_erp(xvect, erp, info, ax=None, sensor_proj=False,
                         xticks=None, xticklabels=None, lw=0.5, title=None,
                         sensor_cols=True, ylabel=None, xtick_skip=1):
    """Plot a GLM-Spectrum contrast with spatial line colouring.

    Parameters
    ----------
    xvect: array_like
        Vector of time values for x-axis
    psd: array_like
        Array of spectrum values to plot
    info: MNE Raw info
        Sensor info for spatial map
    ax: {None or axis handle}
            Axis to plot into (Default value = None)
    sensor_proj: bool
            Whether to plot a topomap inset (Default value = False)
    xticks: array_like
            xtick positions (Default value = None)
    xticklabels: array_like of str  
            xtick labels (Default value = None)
    lw: flot
            Line width(Default value = 0.5)
    title: str  
            Plot title(Default value = None)
    sensor_cols: bool
            Whether to colour lines by sensor (Default value = True)
    ylabel: str
            Y-axis label(Default value = None)
    xtick_skip: int
            Number of xaxis ticks to skip, useful for tight plots (Default value = 1)
    """

    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    plot_sensor_data(xvect, erp, info, ax=ax,
                     sensor_cols=sensor_cols, lw=lw, xticks=xticks,
                     xticklabels=xticklabels, xtick_skip=xtick_skip)
    decorate_spectrum(ax, ylabel=ylabel)
    ax.set_ylim(erp.min())

    if sensor_proj:
        axins = ax.inset_axes([0.6, 0.6, 0.37, 0.37])
        plot_sensor_proj(erp, info, ax=axins)

    if title is not None:
        ax.set_title(title)
        

def plot_sensor_proj(info, ax=None, cmap=None):
    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)
    if np.any(['parcel' in ch for ch in info['ch_names']]):
        parcellation_file = parcellation.guess_parcellation(len(info.ch_names))
        parc_centers = parcellation.parcel_centers(parcellation_file)
        if cmap is None:
            cmap = 'viridis'
            x, y, z = parc_centers.T
            X = y
        else:
            colors = get_source_colors(parcellation_file)
            cmap = ListedColormap(colors)
            X = np.arange(n_parcels)
        
        n_parcels = parc_centers.shape[0]
        plot_markers(
            X,
            parc_centers,
            axes=ax,
            node_size=20,
            node_cmap=cmap,
            annotate=False,
            colorbar=False,
        )
    else:
        plot_channel_layout(ax, info)
    return ax


def plot_sensor_data(xvect, data, info, ax=None, lw=0.5,
                     xticks=None, xticklabels=None,
                     sensor_cols=True, xtick_skip=1):
    """Plot sensor data with spatial line colouring.

    """

    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    if sensor_cols:
        if np.any(['parcel' in ch for ch in info['ch_names']]):
            parcellation_file = parcellation.guess_parcellation(data.T)
            colors = get_source_colors(parcellation_file)
        elif np.any(['state' in ch for ch in info['ch_names']]) or np.any(['mode' in ch for ch in info['ch_names']]):
            colors = None
        else:
            colors, pos, outlines = get_mne_sensor_cols(info)
    else:
        colors = None

    plot_with_cols(ax, data, xvect, colors, lw=lw)
    ax.set_xlim(xvect[0], xvect[-1])

    if xticks is not None:
        ax.set_xticks(xticks[::xtick_skip])
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels[::xtick_skip])
        
        
def get_mne_sensor_cols(info):
    """Get sensor colours from MNE info object.

    Parameters
    ----------
    info : :py:class:`mne.Info <mne.Info>`
        MNE-Python info object

    Returns
    -------
    colors : array_like
        Array of RGB values for each sensor
    pos : array_like
        Sensor positions
    outlines : array_like
        Sensor outlines
    """

    chs = [info['chs'][i] for i in range(len(info['chs']))]
    locs3d = np.array([ch['loc'][:3] for ch in chs])
    x, y, z = locs3d.T
    colors = mne.viz.evoked._rgb(x, y, z)
    pos, outlines = mne.viz.evoked._get_pos_outlines(info,
                                                     range(len(info['chs'])),
                                                     sphere=None)

    return colors, pos, outlines


def decorate_spectrum(ax, ylabel='Amplitude'):
    """Decorate a spectrum plot.
    
    Parameters
    ----------
    ax : :py:class:`matplotlib.axes <matplotlib.axes>`
        Axis to plot into
    ylabel : str
        Y-axis label(Default value = 'Power')
    """
    for tag in ['top', 'right']:
        ax.spines[tag].set_visible(False)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    
    
def get_source_colors(parcellation_file, cmap='viridis'):
    parc_centers = stats.zscore(parcellation.parcel_centers(parcellation_file), axis=0)
    x, y, z = parc_centers.T
    if cmap=='viridis':
        cmap = plt.get_cmap('viridis')
        norm = plt.Normalize(vmin=parc_centers.min(), vmax=parc_centers.max())
        colors = cmap(norm(parc_centers))[:,1,:]
        # colors = colors[np.argsort(y), :]
    else:
        ref = [-5, -5, -3]
        colors = mne.viz.evoked._rgb(x, y, z)
        order = [np.argsort(np.sqrt(ref[i] - parc_centers[:, i]) ** 2) for i in range(3)]
        colors = np.vstack([colors[order[0],0], colors[order[1],1], colors[order[2],2]]).T
    return colors


def plot_channel_layout(ax, info, size=30, marker='o'):
    """Plot sensor layout.

    Parameters
    ----------
    ax : :py:class:`matplotlib.axes <matplotlib.axes>`
        Axis to plot into
    info : :py:class:`mne.Info <mne.Info>`
        MNE-Python info object
    size : int
        Size of sensor § (Default value = 30)
    marker : str
        Marker type (Default value = 'o')
    """

    ax.set_adjustable('box')
    ax.set_aspect('equal')

    colors, pos, outlines = get_mne_sensor_cols(info)
    pos_x, pos_y = pos.T
    mne.viz.evoked._prepare_topomap(pos, ax, check_nonzero=False)
    ax.scatter(pos_x, pos_y,
               color=colors, s=size * .8,
               marker=marker, zorder=1)
    mne.viz.evoked._draw_outlines(ax, outlines)
    
    
def plot_with_cols(ax, data, xvect, cols=None, lw=0.5):
    """Plot data with spatial line colouring.

    Parameters
    ----------
    ax : :py:class:`matplotlib.axes <matplotlib.axes>`
        Axis to plot into
    data : array_like
        Data to plot
    xvect : array_like
        Vector of time values for x-axis
    cols : array_like
        Array of RGB values for each sensor (Default value = None)
    lw : flot
        Line width(Default value = 0.5)
    """
    if cols is not None:
        for ii in range(data.shape[1]):
            ax.plot(xvect, data[:, ii], lw=lw, color=cols[ii, :])
    else:
        ax.plot(xvect, data, lw=lw)
        

def plot_source_topo(
    data_map,
    parcellation_file=None,
    mask_file='MNI152_T1_8mm_brain.nii.gz',
    axis=None,
    cmap=None,
    vmin=None,
    vmax=None,
    alpha=0.7,
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

    Returns
    -------
    image : :py:class:`matplotlib.image.AxesImage <matplotlib.image.AxesImage>`
        AxesImage object
    """
    
    if parcellation_file is None:
        parcellation_file = parcellation.guess_parcellation(data_map)
    parcellation_file = parcellation.find_file(parcellation_file)
    mask_file = parcellation.find_file(mask_file)
    
    if vmin is None:
        vmin = data_map.min()
    if vmax is None:
        vmax = data_map.max()
    
    if vmin < 0 and vmax>0:
        vmax = np.max(np.abs([vmin,vmax]))
        vmin = -vmax
    
    if cmap is None:
        if vmin<0 and vmax>0:
            cmap = 'RdBu_r'
        elif vmin >= 0:
            cmap = 'Reds'
        else:
            cmap = 'Blues_r'
    
    if axis is None:
        # Create figure
        fig, axis = plt.subplots()

    # Fill parcel values into a 3D voxel grid
    data_map = parcellation.parcel_vector_to_voxel_grid(mask_file, parcellation_file, data_map)
    data_map = data_map[..., np.newaxis]
    mask = nib.load(mask_file)
    nii = nib.Nifti1Image(data_map, mask.affine, mask.header)   
    
    # Plot
    plot_glass_brain(
        nii,
        output_file=None,
        display_mode='z',
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