
import pickle
from copy import deepcopy
from pathlib import Path
from itertools import compress

import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.colors import ListedColormap
from scipy import signal, stats

import glmtools as glm
from sails.stft import glm_periodogram
from sails.stft import glm_irasa as sails_glm_irasa

from .glm_base import GLMBaseResult, GroupGLMBaseResult, SensorClusterPerm, SensorMaxStatPerm
from ..source_recon import parcellation

import nibabel as nib
from nilearn.plotting import plot_glass_brain, plot_markers

# --------------------------------------------------------------------------------------------------
# GLM-Spectrum classes designed to work with GLM-Spectra computed from  MNE format sensor space data

class SensorGLMSpectrum(GLMBaseResult):
    """A class for GLM-Spectra fitted from MNE-Python Raw objects."""

    def __init__(self, glmsp, info):
        """
        Parameters
        ----------
        glmsp : :py:class:`glmtools.fit.OLSModel <glmtools.fit.OLSModel>`
            The fitted model object
        info : dict
        """
        self.f = glmsp.f
        self.config = glmsp.config
        super().__init__(glmsp.model, glmsp.design, info, data=glmsp.data)

    def plot_joint_spectrum(self, contrast=0, metric='copes', **kwargs):
        """Plot a GLM-Spectrum contrast with spatial line colouring and topograpies.

        Parameters
        ----------
        contrast : int
             Contrast to plot (Default value = 0)
        freqs : {list, tuple or 'auto'}
             Which frequencies to plot topos for (Default value = 'auto')
        base : float
             The x-axis scaling, set to 0.5 for sqrt freq axis (Default value = 1)
        ax : {None or axis handle}
             Axis to plot into (Default value = None)
        topo_scale : {'joint' or None}
             Whether to fix topomap colour scales across all topos ('joint') or
             leave them individual (Default value = 'joint')
        lw : flot
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
        metric : {'copes' or 'tstats}
             Which metric to plot? (Default value = 'copes')
        """
        if metric == 'copes':
            spec = self.model.copes[contrast, :, :].T
            kwargs['ylabel'] = 'Power' if kwargs.get('ylabel') is None else kwargs.get('ylabel')
        elif metric == 'varcopes':
            spec = self.model.varcopes[contrast, :, :].T
            kwargs['ylabel'] = 'Varcopes' if kwargs.get('ylabel') is None else kwargs.get('ylabel')
        elif metric == 'tstats':
            spec = self.model.tstats[contrast, :, :].T
            kwargs['ylabel'] = 't-statistics' if kwargs.get('ylabel') is None else kwargs.get('ylabel')
        else:
            raise ValueError("Metric '{}' not recognised".format(metric))

        if kwargs.get('title') is None:
            kwargs['title'] = 'C {} : {}'.format(contrast, self.design.contrast_names[contrast])

        plot_joint_spectrum(self.f, spec, self.info, **kwargs)


    def plot_sensor_spectrum(self, contrast=0, metric='copes', **kwargs):
        """Plot a GLM-Spectrum contrast with spatial line colouring.

        Parameters
        ----------
        contrast : int
            Contrast to plot
        sensor_proj : bool
            Whether to plot an inset topo showing spatial colours (Default
            value = False)
        xticks : array_like of float
            xtick positions (Default value = None)
        xticklabels : array_like of str
            labels for xticks (Default value = None)
        ax : {None or axis handle}
            Axis to plot into (Default value = None)
        lw : flot
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
        metric : {'copes' or 'tstats}
            Which metric to plot(Default value = 'copes')

        """
        if metric == 'copes':
            spec = self.model.copes[contrast, :, :].T
            kwargs['ylabel'] = 'Power' if kwargs.get('ylabel') is None else kwargs.get('ylabel')
        elif metric == 'varcopes':
            spec = self.model.varcopes[contrast, :, :].T
            kwargs['ylabel'] = 'Varcopes' if kwargs.get('ylabel') is None else kwargs.get('ylabel')
        elif metric == 'tstats':
            spec = self.model.tstats[contrast, :, :].T
            kwargs['ylabel'] = 't-statistics' if kwargs.get('ylabel') is None else kwargs.get('ylabel')
        else:
            raise ValueError("Metric '{}' not recognised".format(metric))

        if kwargs.get('title') is None:
            kwargs['title'] = 'C {} : {}'.format(contrast, self.design.contrast_names[contrast])

        plot_sensor_spectrum(self.f, spec, self.info, **kwargs)


class GroupSensorGLMSpectrum(GroupGLMBaseResult):
    """A class for group level GLM-Spectra fitted across mmultiple first-level
    GLM-Spectra computed from MNE-Python Raw objects"""

    def __init__(self, model, design, config, info, fl_contrast_names=None, data=None):
        """
        Parameters
        ----------
        model : :py:class:`glmtools.fit.OLSModel <glmtools.fit.OLSModel>'
            The  model object.
        design : :py:class:`glmtools.design.GLMDesign <glmtools.design.GLMDesign>`
            The  GLM design object.
        config : :py:class:`glmtools.config.GLMConfig <glmtools.config.GLMConfig>`
            The  GLM configuration object.
        info : :py:class:`mne.Info <mne.Info>`
            The MNE-Python Info object.
        fl_contrast_names : {None, list}
            List of first-level contrast names (Default value = None)
        data : :py:class:`glmtools.data.TrialGLMData <glmtools.data.TrialGLMData>`
            The data object used to fit the model (Default value = None)
        """
            
            
        self.f = config.freqvals
        super().__init__(model, design, info, config, fl_contrast_names=fl_contrast_names, data=data)

    def __str__(self):
        msg = 'GroupSensorGLMSpectrum\n'
        line = '\tData - {} Inputs, {} Channels and {} Frequencies\n'
        msg += line.format(self.design.design_matrix.shape[0], self.model.copes.shape[2], self.model.copes.shape[3])

        line = '\tFirst-Level - {} Regressors and {} Contrasts\n'
        msg += line.format(self.design.design_matrix.shape[1], self.model.copes.shape[1])

        line = '\tGroup-Level - {} Regressors and {} Contrasts\n'
        msg += line.format(self.design.design_matrix.shape[1], self.model.copes.shape[0])
        return msg

    def save_pkl(self, outname, overwrite=True, save_data=False):
        """Save GLM-Spectrum result to a pickle file.

        Parameters
        ----------
        outname : str
             Filename or full file path to write pickle to
        overwrite : bool
             Overwrite previous file if one exists? (Default value = True)
        save_data : bool
             Save STFT data in pickle? This is omitted by default to save disk
             space (Default value = False)
        """
        if Path(outname).exists() and not overwrite:
            msg = "{} already exists. Please delete or do use overwrite=True."
            raise ValueError(msg.format(outname))


        self.config.detrend_func = None  # Have to drop this to pickle

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

    def plot_joint_spectrum(self, gcontrast=0, fcontrast=0, metric='copes', **kwargs):
        """Plot a GLM-Spectrum contrast with spatial line colouring and topograpies.

        Parameters
        ----------
        gcontrast : int
             Group level contrast to plot (Default value = 0)
        fcontrast : int
             First level contrast to plot (Default value = 0)
        freqs : {list, tuple or 'auto'}
             Which frequencies to plot topos for (Default value = 'auto')
        base : float
             The x-axis scaling, set to 0.5 for sqrt freq axis (Default value = 1)
        ax : {None or axis handle}
             Axis to plot into (Default value = None)
        topo_scale : {'joint' or None}
             Whether to fix topomap colour scales across all topos ('joint') or
             leave them individual (Default value = 'joint')
        lw : flot
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
        metric : {'copes' or 'tstats}
             Which metric to plot(Default value = 'copes')
        """
        if metric == 'copes':
            spec = self.model.copes[gcontrast, fcontrast, :, :].T
            kwargs['ylabel'] = 'Power' if kwargs.get('ylabel') is None else kwargs.get('ylabel')
        elif metric == 'varcopes':
            spec = self.model.varcopes[gcontrast, fcontrast, :, :].T
            kwargs['ylabel'] = 'Varcopes' if kwargs.get('ylabel') is None else kwargs.get('ylabel')
        elif metric == 'tstats':
            spec = self.model.tstats[gcontrast, fcontrast, :, :].T
            kwargs['ylabel'] = 't-statistics' if kwargs.get('ylabel') is None else kwargs.get('ylabel')
        else:
            raise ValueError("Metric '{}' not recognised".format(metric))

        if kwargs.get('title') is None:
            gtitle = 'group con : {}'.format(self.contrast_names[gcontrast])
            ftitle = 'first-level con : {}'.format(self.fl_contrast_names[fcontrast])

            kwargs['title'] = gtitle + '\n' + ftitle

        plot_joint_spectrum(self.f, spec, self.info, **kwargs)


    def get_fl_contrast(self, fl_con):
        """Get the data from a single first level contrast.

        Parameters
        ----------
        fl_con : int
            First level contrast data index to return

        Returns
        -------
        ret_con : :py:class:`GroupSensorGLMSpectrum <osl_ephys.glm.glm_spectrum.GroupSensorGLMSpectrum>`
            GroupSensorGLMSpectrum instance containing a single first level contrast.
        """
        ret_con = deepcopy(self.data)
        ret_con.data = ret_con.data[:, fl_con, :, :]

        return ret_con


class MaxStatPermuteGLMSpectrum(SensorMaxStatPerm):
    """A class holding the result for sensor x frequency cluster stats computed
    from a group level GLM-Spectrum"""

    def plot_sig_clusters(self, thresh, ax=None, base=1, min_extent=1):
        """Plot the significant clusters at a given threshold.

        Parameters
        ----------
        thresh : float
            The threshold to consider a cluster significant eg 95 or 99
        ax : :py:class:`matplotlib.axes <matplotlib.axes>`
            Matplotlib axes to plot on. (Default value = None)
        base : float
            The x-axis scaling, set to 0.5 for sqrt freq axis (Default value = 1)
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

        plot_joint_spectrum_clusters(self.f, obs, clu, self.info, base=base, ax=ax, title=title, ylabel='t-stat')


class ClusterPermuteGLMSpectrum(SensorClusterPerm):
    """A class holding the result for sensor x frequency cluster stats computed
    from a group level GLM-Spectrum"""

    def plot_sig_clusters(self, thresh, ax=None, base=1, min_extent=1):
        """Plot the significant clusters at a given threshold.

        Parameters
        ----------
        thresh : float
            The threshold to consider a cluster significant eg 95 or 99
        ax : :py:class:`matplotlib.axes <matplotlib.axes>`
            Matplotlib axes to plot on. (Default value = None)
        base : float
             The x-axis scaling, set to 0.5 for sqrt freq axis (Default value = 1). (Default value = 1)

        """
        title = 'group-con: {}\nfirst-level-con: {}'
        title = title.format(self.gl_contrast_name, self.fl_contrast_name)

        clu, obs_sel = self.perms.get_sig_clusters(thresh, self.perm_data) # obs here is the selected data. We want to plot the full data
        obs = glm.fit.OLSModel(self.perms._design, self.perm_data)
        obs = obs.get_tstats(**self.perms.tstat_args)[self.gl_con, :, :]
        
        to_plot = []
        for c in clu:
            to_plot.append(False if len(c[2][0]) < min_extent or len(c[2][1]) < min_extent else True)
        clu = list(compress(clu, to_plot))

        plot_joint_spectrum_clusters(self.f, obs, clu, self.info, base=base, ax=ax, title=title, ylabel='t-stat')


# ------------------------------------------------------------------------------
# GLM-Spectrum helper functions that take raw objects (or numpy arrays with MNE
# dim-order), compute GLM-Spectra and return class objects containing the result

def group_glm_spectrum(inspectra, design_config=None, datainfo=None, metric='copes'):
    """Compute a group GLM-Spectrum from a set of first-level GLM-Spectra.

    Parameters
    ----------
    inspectra : list, tuple
        A list containing either the filepaths of a set of saved GLM-Spectra
        objects, or the GLM-Spectra objects themselves.
    design_config : :py:class:`glmtools.design.DesignConfig <glmtools.design.DesignConfig>`
         The design specification for the group level model (Default value = None)
    datainfo : dict
         Dictionary of data values to use as covariates. The length of each
         covariate must match the number of input GLM-Spectra (Default value =
         None)
    metric : {'copes', or 'tsats'}
         Which metric to plot (Default value = 'copes')

    Returns
    -------
    :py:class:`GroupSensorGLMSpectrum <osl_ephys.glm.GroupSensorGLMSpectrum>`
        GroupSensorGLMSpectrum instance containing the group level GLM-Spectrum.

    References
    ----------
    .. [1] Quinn, A. J., Atkinson, L., Gohil, C., Kohl, O., Pitt, J., Zich, C., Nobre,
       A. C., & Woolrich, M. W. (2022). The GLM-Spectrum: A multilevel framework
       for spectrum analysis with covariate and confound modelling. Cold Spring
       Harbor Laboratory. https://doi.org/10.1101/2022.11.14.516449
    """
    datainfo = {} if datainfo is None else datainfo

    fl_data = []
    ## Need to sanity check that info and configs match before concatenating
    for ii in range(len(inspectra)):
        if isinstance(inspectra[ii], str):
            glmsp = read_glm_spectrum(inspectra[ii])
        else:
            glmsp = inspectra[ii]

        fl_data.append(getattr(glmsp.model, metric)[np.newaxis, ...])
        fl_contrast_names = glmsp.design.contrast_names

    fl_data = np.concatenate(fl_data, axis=0)
    group_data = glm.data.TrialGLMData(data=fl_data, **datainfo)

    if design_config is None:
        design_config = glm.design.DesignConfig()
        design_config.add_regressor(name='Mean', rtype='Constant')
        design_config.add_simple_contrasts()

    design = design_config.design_from_datainfo(group_data.info)
    model = glm.fit.OLSModel(design, group_data)

    return GroupSensorGLMSpectrum(model, design, glmsp.config, glmsp.info, data=group_data, fl_contrast_names=fl_contrast_names)


def glm_spectrum(XX, reg_categorical=None, reg_ztrans=None, reg_unitmax=None,
                 contrasts=None, fit_intercept=True, standardise_data=False,
                 window_type='hann', nperseg=None, noverlap=None, nfft=None,
                 detrend='constant', return_onesided=True, scaling='density',
                 mode='psd', fmin=None, fmax=None, axis=-1, fs=1, verbose='WARNING'):
    """Compute a GLM-Spectrum from a MNE-Python Raw data object.

    Parameters
    ----------
    XX : {MNE Raw object, or data array}
        Data to compute GLM-Spectrum from
    standardise_data : bool
        Flag indicating whether to z-transform input data (Default value = False)
    reg_categorical : dict or None
        Dictionary of covariate time series to be added as binary regessors. (Default value = None)
    reg_ztrans : dict or None
        Dictionary of covariate time series to be added as z-standardised regessors. (Default value = None)
    reg_unitmax : dict or None
        Dictionary of confound time series to be added as positive-valued unitmax regessors. (Default value = None)
    contrasts : dict or None
        Dictionary of contrasts to be computed in the model.
        (Default value = None, will add a simple contrast for each regressor)
    fit_intercept : bool
        Specifies whether a constant valued 'intercept' regressor is included in the model. (Default value = True)'

    nperseg : int
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int
        Number of samples that successive sliding windows should overlap.
    window_type : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `scipy.signal.windows.get_window` to generate the
        window values, which are DFT-even by default. See `scipy.signal.windows`
        for a list of windows and required parameters.
        If `window` is array_like it will be used directly as the window and its
        length must be nperseg. Defaults to a Hann window.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.'

    nfft : int
        Length of the FFT to use (Default value = 256)
    axis : int
        Axis of input array along which the computation is performed. (Default value = -1)
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    mode : {'psd', 'magnitude', 'angle', 'phase', 'complex'}
        Which type of spectrum to return (Default value = 'psd')
    scaling : { 'density', 'spectrum' }
        Selects between computing the power spectral density ('density')
        where `Pxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'
    fs : float
        Sampling rate of the data
    fmin : {float, None}
        Smallest frequency in desired range (left hand boundary)
    fmax : {float, None}
        Largest frequency in desired range (right hand boundary)'

    verbose : {None, 'DEBUG', 'INFO', 'WARNING', 'CRITICAL'}
        String indicating the level of detail to be printed to the screen during computation.'

    Returns
    -------
    :py:class:`SensorGLMSpectrum <osl_ephys.glm.glm_spectrum.SensorGLMSpectrum>`
        SensorGLMSpectrum instance containing the fitted GLM-Spectrum.

    References
    ----------
    .. [1] Quinn, A. J., Atkinson, L., Gohil, C., Kohl, O., Pitt, J., Zich, C., Nobre,
       A. C., & Woolrich, M. W. (2022). The GLM-Spectrum: A multilevel framework
       for spectrum analysis with covariate and confound modelling. Cold Spring
       Harbor Laboratory. https://doi.org/10.1101/2022.11.14.516449
    """
    if isinstance(XX, mne.io.base.BaseRaw):
        fs = XX.info['sfreq']
        nperseg = int(np.floor(fs)) if nperseg is None else nperseg
        YY = XX.get_data()
        axis = 1
    else:
        YY = XX

    if standardise_data:
        YY = stats.zscore(YY, axis=axis)

    # sails.sftf.config freqvals isn't right when frange is trimmed!
    glmsp = glm_periodogram(YY, axis=axis,
                            reg_categorical=reg_categorical,
                            reg_ztrans=reg_ztrans,
                            reg_unitmax=reg_unitmax,
                            contrasts=contrasts,
                            fit_intercept=fit_intercept,
                            window_type=window_type,
                            fs=fs,
                            nperseg=nperseg,
                            noverlap=noverlap,
                            nfft=nfft,
                            detrend=detrend,
                            return_onesided=return_onesided,
                            scaling=scaling,
                            mode=mode,
                            fmin=fmin,
                            fmax=fmax,
                            verbose=verbose)

    if isinstance(XX, mne.io.base.BaseRaw):
        return SensorGLMSpectrum(glmsp, XX.info)
    else:
        return glmsp


def glm_irasa(XX, method='modified', resample_factors=None, aperiodic_average='median',
                 reg_categorical=None, reg_ztrans=None, reg_unitmax=None,
                 contrasts=None, fit_intercept=True, standardise_data=False,
                 window_type='hann', nperseg=None, noverlap=None, nfft=None,
                 detrend='constant', return_onesided=True, scaling='density',
                 mode='psd', fmin=None, fmax=None, axis=-1, fs=1, verbose='WARNING'):
    """Compute a GLM-IRASA from a MNE-Python Raw data object.

    Parameters
    ----------
    XX : {MNE Raw object, or data array}
        Data to compute GLM-Spectrum from
    standardise_data : bool
        Flag indicating whether to z-transform input data (Default value = False)
    reg_categorical : dict or None
        Dictionary of covariate time series to be added as binary regessors. (Default value = None)
    reg_ztrans : dict or None
        Dictionary of covariate time series to be added as z-standardised regessors. (Default value = None)
    reg_unitmax : dict or None
        Dictionary of confound time series to be added as positive-valued unitmax regessors. (Default value = None)
    contrasts : dict or None
        Dictionary of contrasts to be computed in the model.
        (Default value = None, will add a simple contrast for each regressor)
    fit_intercept : bool
        Specifies whether a constant valued 'intercept' regressor is included in the model. (Default value = True)'

    method : {'original', 'modified'}
        whether to compute the original implementation of IRASA or the modified update
        (default is 'modified')
    resample_factors : {None, array_like}
        array of resampling factors to average across or None, in which a set
        of factors are automatically computed (default is None).
    aperiodic_average : {'mean', 'median', 'median_bias', 'min'}
        method for averaging across irregularly resampled spectra to estimate
        the aperiodic component (default is 'median').'

    nperseg : int
        Length of each segment. Defaults to None, but if window is str or
        tuple, is set to 256, and if window is array_like, is set to the
        length of the window.
    noverlap : int
        Number of samples that successive sliding windows should overlap.
    window_type : str or tuple or array_like, optional
        Desired window to use. If `window` is a string or tuple, it is
        passed to `scipy.signal.windows.get_window` to generate the
        window values, which are DFT-even by default. See `scipy.signal.windows`
        for a list of windows and required parameters.
        If `window` is array_like it will be used directly as the window and its
        length must be nperseg. Defaults to a Hann window.
    detrend : str or function or `False`, optional
        Specifies how to detrend each segment. If `detrend` is a
        string, it is passed as the `type` argument to the `detrend`
        function. If it is a function, it takes a segment and returns a
        detrended segment. If `detrend` is `False`, no detrending is
        done. Defaults to 'constant'.'

    nfft : int
        Length of the FFT to use (Default value = 256)
    axis : int
        Axis of input array along which the computation is performed. (Default value = -1)
    return_onesided : bool, optional
        If `True`, return a one-sided spectrum for real data. If
        `False` return a two-sided spectrum. Defaults to `True`, but for
        complex data, a two-sided spectrum is always returned.
    mode : {'psd', 'magnitude', 'angle', 'phase', 'complex'}
        Which type of spectrum to return (Default value = 'psd')
    scaling : { 'density', 'spectrum' }
        Selects between computing the power spectral density ('density')
        where `Pxx` has units of V**2/Hz and computing the power
        spectrum ('spectrum') where `Pxx` has units of V**2, if `x`
        is measured in V and `fs` is measured in Hz. Defaults to
        'density'
    fs : float
        Sampling rate of the data
    fmin : {float, None}
        Smallest frequency in desired range (left hand boundary)
    fmax : {float, None}
        Largest frequency in desired range (right hand boundary)'

    verbose : {None, 'DEBUG', 'INFO', 'WARNING', 'CRITICAL'}
        String indicating the level of detail to be printed to the screen during computation.'

    Returns
    -------
    :py:class:`SensorGLMSpectrum <osl_ephys.glm.glm_spectrum.SensorGLMSpectrum>`
        SensorGLMSpectrum instance containing the fitted GLM-Spectrum.
        
    References
    ----------
    .. [1] Quinn, A. J., Atkinson, L., Gohil, C., Kohl, O., Pitt, J., Zich, C., Nobre,
       A. C., & Woolrich, M. W. (2022). The GLM-Spectrum: A multilevel framework
       for spectrum analysis with covariate and confound modelling. Cold Spring
       Harbor Laboratory. https://doi.org/10.1101/2022.11.14.516449

    """
    if isinstance(XX, mne.io.base.BaseRaw):
        fs = XX.info['sfreq']
        nperseg = int(np.floor(fs)) if nperseg is None else nperseg
        YY = XX.get_data()
        axis = 1
    else:
        YY = XX

    if standardise_data:
        YY = stats.zscore(YY, axis=axis)

    # sails.sftf.config freqvals isn't right when frange is trimmed!
    aper, osc = sails_glm_irasa(YY, axis=axis,
                            method=method,
                            resample_factors=resample_factors,
                            aperiodic_average=aperiodic_average,
                            reg_categorical=reg_categorical,
                            reg_ztrans=reg_ztrans,
                            reg_unitmax=reg_unitmax,
                            contrasts=contrasts,
                            fit_intercept=fit_intercept,
                            window_type=window_type,
                            fs=fs,
                            nperseg=nperseg,
                            noverlap=noverlap,
                            nfft=nfft,
                            detrend=detrend,
                            return_onesided=return_onesided,
                            scaling=scaling,
                            mode=mode,
                            fmin=fmin,
                            fmax=fmax,
                            verbose=verbose)

    if isinstance(XX, mne.io.base.BaseRaw):
        return SensorGLMSpectrum(aper, XX.info), SensorGLMSpectrum(osc, XX.info)
    else:
        return aper, osc


def read_glm_spectrum(infile):
    """Read in a GLMSpectrum object that has been saved as as a pickle.

    Parameters
    ----------
    infile : str
        Filepath of saved object

    Returns
    -------
    glmsp: :py:class:`SensorGLMSpectrum <osl_ephys.glm.glm_spectrum.SensorGLMSpectrum>`
        SensorGLMSpectrum instance containing the fitted GLM-Spectrum.
    """
    with open(infile, 'rb') as outp:
        glmsp = pickle.load(outp)
    return glmsp

# --------
# Plotting


def plot_joint_spectrum_clusters(xvect, psd, clusters, info, ax=None, freqs='auto', base=1,
                                 topo_scale='joint', lw=0.5, ylabel='Power', title='', ylim=None,
                                 xtick_skip=1, topo_prop=1/5, topo_cmap=None, topomap_args=None):
    """Plot a GLM-Spectrum contrast from cluster objects, with spatial line colouring and topograpies.

    Parameters
    ----------
    xvect : array_like
        Frequency vector
    psd : array_like
        Spectrum values
    clusters : list
        List of cluster objects
    info : dict 
        MNE-Python info object
    ax : {None or axis handle}
        Axis to plot into (Default value = None)
    freqs : {list, tuple or 'auto'}
        Which frequencies to plot topos for (Default value = 'auto')
    base : float
        The x-axis scaling, set to 0.5 for sqrt freq axis (Default value = 1)
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
    
    plot_sensor_spectrum(xvect, psd, info, ax=main_ax, base=base, lw=0.25, ylabel=ylabel)
    fx = prep_scaled_freq(base, xvect)

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

    # Reorder clusters in ascending frequency
    clu_order = []
    for clu in clusters:
        clu_order.append(clu[2][0].min())
    clusters = [clusters[ii] for ii in np.argsort(clu_order)]

    print('\n')
    table_header = '{0:12s}{1:16s}{2:12s}{3:12s}{4:14s}'
    print(table_header.format('Cluster', 'Statistic', 'Freq Min', 'Freq Max', 'Num Channels'))
    table_template = '{0:<12d}{1:<16.3f}{2:<12.2f}{3:<12.2f}{4:<14d}'

    if type(freqs)==str and freqs=='auto':
        topo_centres = np.linspace(0, 1, len(clusters)+2)[1:-1]
        freqs_topo = freqs
    else:
        topo_centres = np.linspace(0, 1, len(freqs)+2)[1:-1]
        freqs_topo = list(np.sort(freqs.copy()))
    topo_width = 0.4
    topos = []
    ymax_span = (np.abs(yl[0]) + yl[1]) / (np.abs(yl[0]) + yl[1]*1.2)
    
    data_toplot = []
    topo_ax_toplot = []
    for idx in range(len(clusters)):
        clu = clusters[idx]

        # Extract cluster location in space and frequency
        channels = np.zeros((psd.shape[1], ))
        channels[clu[2][1]] = 1
        if len(channels) == 204:
            channels = np.logical_or(channels[::2], channels[1::2])
        freqs = np.zeros((psd.shape[0], ))
        freqs[clu[2][0]] = 1
        finds = np.where(freqs)[0]
        if len(finds) == 1:
            if finds[0]<len(fx[0])-1: 
                finds = np.array([finds[0], finds[0]+1])
            else: # can't extend to next freq if last freq
                finds = np.array([finds[0], finds[0]])

        msg = 'Cluster {} - stat: {}, freq range: {}, num channels {}'
        freq_range = (fx[0][finds[0]], fx[0][finds[-1]])
        print(table_template.format(idx+1, clu[0], freq_range[0], freq_range[1], int(channels.sum())))

        # Plot cluster span overlay on spectrum
        main_ax.axvspan(fx[0][finds[0]], fx[0][finds[-1]], facecolor=[0.7, 0.7, 0.7], alpha=0.5, ymax=ymax_span)
        if type(freqs_topo)==str and freqs_topo=='auto':
            fmid = int(np.floor(finds.mean()))
            # Create topomap axis
            topo_pos = [topo_centres[idx] - 0.2, 1-title_prop-topo_prop, 0.4, topo_prop]
        else:
            fmid_tmp = np.where([ifreq<=xvect[finds[-1]] and ifreq>xvect[finds[0]] for ifreq in freqs_topo])[0]
            if len(fmid_tmp)>0:
                fmid = np.argmin(np.abs(xvect - freqs_topo[fmid_tmp[0]]))
                # Create topomap axis
                topo_pos = [topo_centres[len(topo_centres)-len(freqs_topo)] - 0.2, 1-title_prop-topo_prop, 0.4, topo_prop]
                
                freqs_topo.pop(fmid_tmp[0])
            else:
                continue
        
        # Create topomap axis
        topo_ax = ax.inset_axes(topo_pos)

        # Plot connecting line to topo
        xy_main = (fx[0][fmid], yl[1])
        xy_topo = (0.5, 0)
        con = ConnectionPatch(
            xyA=xy_main, coordsA=main_ax.transData,
            xyB=xy_topo, coordsB=topo_ax.transAxes,
            arrowstyle="-", color=[0.7, 0.7, 0.7])
        main_ax.figure.add_artist(con)

        # save the topo axis and data for later
        data_toplot.append(psd[fmid, :])
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
                im = parcellation.plot_source_topo(topo, axis=topo_ax, cmap=topo_cmap) 
            else:
                im, cn = mne.viz.plot_topomap(topo, info, axes=topo_ax, show=False, mask=channels, ch_type='planar1', cmap=topo_cmap)
            im.set_clim(vmin, vmax)
            topos.append(im)

        cb_pos = [0.95, 1-title_prop-topo_prop, 0.025, topo_prop]
        cax =  ax.inset_axes(cb_pos)
        plt.colorbar(topos[0], cax=cax)

    print('\n')  # End table

    ax.set_title(title, x=0.5, y=1-title_prop)
    

def plot_joint_spectrum(xvect, psd, info, ax=None, freqs='auto', base=1,
        topo_scale='joint', lw=0.5, ylabel='Power', title='', ylim=None,
        xtick_skip=1, topo_prop=1/5, topo_cmap=None, topomap_args=None):
    """Plot a GLM-Spectrum contrast with spatial line colouring and topograpies.

    Parameters
    ----------
    xvect : array_like
        Vector of frequency values for x-axis
    psd : array_like
        Array of spectrum values to plot
    info : :py:class:`mne.Info <mne.Info>`
        MNE-Python info object
    ax : {None or axis handle}
         Axis to plot into (Default value = None)
    freqs : {list, tuple or 'auto'}
         Which frequencies to plot topos for (Default value = 'auto')
    base : float
         The x-axis scaling, set to 0.5 for sqrt freq axis (Default value = 1)
    topo_scale : {'joint' or None}
         Whether to fix topomap colour scales across all topos ('joint') or
         leave them individual (Default value = 'joint')
    lw : flot
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
    topo_cmap : {None or matplotlib colormap}
        Colormap to use for plotting (Default value is 'RdBu_r' if pooled topo data range
        is positive and negative, otherwise 'Reds' or 'Blues' depending on sign of
        pooled data range)
    """
    if ax is None:
        fig = plt.figure()
        fig.subplots_adjust(top=0.8)
        ax = plt.subplot(111)

    topomap_args = {} if topomap_args is None else topomap_args

    ax.set_axis_off()

    title_prop = 0.1
    main_prop = 1-title_prop-topo_prop
    main_ax = ax.inset_axes((0, 0, 1, main_prop))

    plot_sensor_spectrum(xvect, psd, info, ax=main_ax, base=base, lw=0.25, ylabel=ylabel)
    fx = prep_scaled_freq(base, xvect)

    if freqs == 'auto':
        if base == 1:
            topo_freq_inds = signal.find_peaks(np.abs(psd.mean(axis=1)), distance=xvect.shape[0]/3)[0]
            if len(topo_freq_inds) > 2:
                I = np.argsort(np.abs(psd.mean(axis=1))[topo_freq_inds])[-2:]
                topo_freq_inds = topo_freq_inds[I]
        elif base == 0.5: # using distance is a bit tricky with sqrt freqs
            dist = xvect.shape[0]/2.5
            tmp_topo_freq_inds = signal.find_peaks(np.abs(psd.mean(axis=1)))[0]
            topo_freq_inds = []
            for i, ifrq in enumerate(tmp_topo_freq_inds):
                if i==0:
                    topo_freq_inds.append(ifrq)
                elif len(topo_freq_inds)==3:
                    continue
                elif (np.argmin(np.abs(np.sqrt(ifrq)-fx[0])) - np.argmin(np.abs(np.sqrt(tmp_topo_freq_inds[i-1])-fx[0]))) < dist:
                    topo_freq_inds.append(ifrq)
            topo_freq_inds = np.array(topo_freq_inds)
        freqs = xvect[topo_freq_inds]
    else:
        topo_freq_inds = [np.argmin(np.abs(xvect - ff)) for ff in freqs]

    yl = main_ax.get_ylim() if ylim is None else ylim
    yfactor = 1.2 if yl[1] > 0 else 0.8
    main_ax.set_ylim(yl[0], yfactor*yl[1])
    #yl = main_ax.get_ylim()

    yt = ax.get_yticks()
    inds = yt < yl[1]
    ax.set_yticks(yt[inds])

    ax.figure.canvas.draw()
    offset = ax.yaxis.get_major_formatter().get_offset()
    ax.yaxis.offsetText.set_visible(False)
    ax.text(0, yl[1], offset, ha='right')

    # determine colorbar
    if topo_cmap is None:
        if psd[topo_freq_inds, :].min() < 0 and psd[topo_freq_inds, :].max() > 0:
            topo_cmap = 'RdBu_r'
        elif psd[topo_freq_inds, :].min() >= 0:
            topo_cmap = 'Reds'
        else:
            topo_cmap = 'Blues_r'
    
    topo_centres = np.linspace(0, 1, len(freqs)+2)[1:-1]
    topo_width = 0.4
    topos = []
    dats=[]
    for idx in range(len(freqs)):
        # Create topomap axis
        topo_pos = [topo_centres[idx] - 0.2, 1-title_prop-topo_prop, 0.4, topo_prop]
        topo_ax = ax.inset_axes(topo_pos)

        topo_idx = fx[0][topo_freq_inds[idx]]
        main_ax.plot((topo_idx, topo_idx), yl, color=[0.7, 0.7, 0.7])

        xy_main = (topo_idx, yl[1])
        xy_topo = (0.5, 0)
        con = ConnectionPatch(
            xyA=xy_main, coordsA=main_ax.transData,
            xyB=xy_topo, coordsB=topo_ax.transAxes,
            arrowstyle="-", color=[0.7, 0.7, 0.7])
        ax.figure.add_artist(con)

        dat = psd[topo_freq_inds[idx], :]
        if np.any(['parcel' in ch for ch in info['ch_names']]): # source data
            im = parcellation.plot_source_topo(dat, axis=topo_ax, cmap=topo_cmap)
        else:
            im, cn = mne.viz.plot_topomap(dat, info, axes=topo_ax, show=False, cmap=topo_cmap)
        topos.append(im)
        dats.append(dat)
        
    if topo_scale == 'joint':
        vmin = np.min([t.min() for t in dats])
        vmax = np.max([t.max() for t in dats])
        if vmin < 0 and vmax > 0:
            vmax = np.max(np.abs([vmin,vmax]))
            vmin = -vmax
        for t in topos:
            t.set_clim(vmin, vmax)
    else:
        vmin = 0
        vmax = 1

    cb_pos = [0.95, 1-title_prop-topo_prop, 0.025, topo_prop]
    cax =  ax.inset_axes(cb_pos)

    plt.colorbar(topos[0], cax=cax)

    ax.set_title(title, x=0.5, y=1-title_prop)


def plot_sensor_spectrum(xvect, psd, info, ax=None, sensor_proj=False,
                         xticks=None, xticklabels=None, lw=0.5, title=None,
                         sensor_cols=True, base=1, ylabel=None, xtick_skip=1):
    """Plot a GLM-Spectrum contrast with spatial line colouring.

    Parameters
    ----------
    xvect: array_like
        Vector of frequency values for x-axis
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
    base: float
            The x-axis scaling, set to 0.5 for sqrt freq axis (Default value = 1)
    ylabel: str
            Y-axis label(Default value = None)
    xtick_skip: int
            Number of xaxis ticks to skip, useful for tight plots (Default value = 1)
    """

    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    plot_sensor_data(xvect, psd, info, ax=ax, base=base,
                     sensor_cols=sensor_cols, lw=lw, xticks=xticks,
                     xticklabels=xticklabels, xtick_skip=xtick_skip)
    decorate_spectrum(ax, ylabel=ylabel)
    ax.set_ylim(psd.min())

    if sensor_proj:
        axins = ax.inset_axes([0.6, 0.6, 0.37, 0.37])
        plot_sensor_proj(info, ax=axins)

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
                     sensor_cols=True, base=1, xtick_skip=1):
    """Plot sensor data with spatial line colouring.

    """

    if ax is None:
        fig = plt.figure()
        ax = plt.subplot(111)

    fx, xticklabels, xticks = prep_scaled_freq(base, xvect)

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

    plot_with_cols(ax, data, fx, colors, lw=lw)
    ax.set_xlim(fx[0], fx[-1])

    if xticks is not None:
        ax.set_xticks(xticks[::xtick_skip])
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels[::xtick_skip])


def prep_scaled_freq(base, freq_vect):
    """Prepare frequency vector for plotting with a given scaling.

    Parameters
    ----------
    base : float
        The x-axis scaling, set to 0.5 for sqrt freq axis (Default value = 1)  
    freq_vect : array_like
        Vector of frequency values for x-axis
        
    Returns
    -------
    fx : array_like
        Scaled frequency vector
    ftick : array_like
        Normal frequency ticks
    ftickscaled : array_like
        Scaled frequency ticks

    Notes
    -----
        Assuming ephy freq ranges for now - around 1-40Hz
    """
    fx = freq_vect**base
    if base < 1:
        nticks = int(np.floor(np.sqrt(freq_vect[-1])))
        ftick = np.array([ii**2 for ii in range(1,nticks+1)])
        ftickscaled = ftick**base
    else:
        # Stick with automatic scales
        ftick = None
        ftickscaled = None
    return fx, ftick, ftickscaled


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
        Vector of frequency values for x-axis
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


def decorate_spectrum(ax, ylabel='Power'):
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
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel(ylabel)
