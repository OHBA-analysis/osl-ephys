import numpy as np
from scipy.stats import kurtosis
from functools import partial
import os

import matplotlib.pyplot as plt

# mean_psd_in_band lives in osl_ephys.utils (shared with the manual_ica
# subpackage). Re-exported here so semp code / configs can keep importing it
# from semp.metric. It is NOT interchangeable with psd_band_stat below -- see
# that function's docstring for the difference.
from osl_ephys.utils import mean_psd_in_band  # noqa: F401

EEG_BANDS = {
    'alpha': (8, 12),
    'beta': (13, 30),
    'gamma': (30, 100),
    'delta': (0.5, 4),
    'theta': (4, 8)
}

class EEGTracer:
    """A class to trace the evolution of EEG signal statistics across preprocessing steps.

    Attributes:
        checkpoints (list): A list of dicts containing checkpoint names and computed statistics.
        functions (dict): A mapping of statistic names to functions that compute them.
    """
    
    def __init__(self, **kwargs):
        self.checkpoints = []
        
        # Never use lambda functions directly in the class, as they cannot be pickled.
        # Instead, define functions or use partial functions.
        self.functions = {
            # 'variance': lambda x: np.var(x, axis=1),  # this is not picklable
            'variance': partial(np.var, axis=1),
        }
        self.psd_functions = {}
        for key, value in kwargs.items():
            if not callable(value):
                raise ValueError(f"Function {key} must be callable.")
            if key.startswith('psd_'):
                self.psd_functions[key] = value
            else:
                self.functions[key] = value
        
    def checkpoint(self, data, name):
        """Store the variance of the data at a checkpoint.
        
        Args:
            data (np.ndarray): The data to check. Should be a 2D array (#ch, #timepoints).
            name (str): The name of the checkpoint.
        """
        
        ckpt = {}
        ckpt['name'] = name
        
        for func_name, func in self.functions.items():
            ckpt[func_name] = func(data)
        
        self.checkpoints.append(ckpt)
    
    def checkpoint_psd(self, psd, name):
        """Store the PSD of the data at a checkpoint.
        
        Args:
            psd (mne.time_frequency.Spectrum): The PSD data to check.
            name (str): The name of the checkpoint.
        """
        psd_ckpt = {}
        psd_ckpt['name'] = name
        
        for func_name, func in self.psd_functions.items():
            psd_ckpt[func_name] = func(psd)
        
        self.checkpoints[-1].update(psd_ckpt)
        
    def plot(self, func_list=None, scalarizer=np.mean, figsize=(10,3), save_pth=None, show=True):
        if not self.checkpoints:
            raise RuntimeError("No checkpoints recorded. Call .checkpoint() before plotting.")
        
        if func_list is None:
            functions = list(self.functions.keys()) + list(self.psd_functions.keys())
        else:
            functions = func_list

        if callable(scalarizer):
            scalarizers = {fn: scalarizer for fn in functions}
        elif isinstance(scalarizer, dict):
            scalarizers = {fn: scalarizer.get(fn, np.mean) for fn in functions}
        else:
            raise ValueError("scalarizer must be callable or dict, with keys matching function names.")

        checkpoint_names = [ckpt['name'] for ckpt in self.checkpoints]
        x_positions = range(len(checkpoint_names))

        markers = ['o', 's', '^', 'D', 'x']
        colors = plt.cm.tab10.colors  # default Matplotlib qualitative palette
        
        for idx, func_name in enumerate(functions):
            values = []
            for ckpt in self.checkpoints:
                if func_name in ckpt:
                    values.append(scalarizers[func_name](ckpt[func_name]))
                else:
                    values.append(np.nan)  # placeholder for missing metric
        
        
            for yscale in ['linear', 'log']:
                fig, ax = plt.subplots(figsize=figsize)
                ax.plot(
                    x_positions, values,
                    marker=markers[idx % len(markers)],
                    color=colors[idx % len(colors)]
                )
                ax.set_ylabel(func_name.capitalize())
                ax.set_title(f"{func_name.capitalize()} at Checkpoints ({yscale})")
                ax.grid(True)
                ax.set_yscale(yscale)

                ax.set_xticks(x_positions)
                ax.set_xticklabels(checkpoint_names, rotation=30, ha="right")
                ax.set_xlabel("Checkpoint")

                plt.tight_layout()

                if save_pth:
                    outdir = save_pth / "tracer"
                    outdir.mkdir(parents=True, exist_ok=True)
                    outname = f"{func_name}_{yscale}.png"
                    plt.savefig(outdir / outname, dpi=300)

                if show:
                    plt.show()
                plt.close(fig)


def psd_band_ratio(psd, band1, fn1, band2=None, fn2=None):
    """Calculate the ratio of the statistic in band1 to the statistic in band2.

    Args:
        psd (Spectrum): The power spectral density data.
        band1 (list or str): band range of the first band (fmin, fmax) on the numerator. if str, it should be a key in EEG_BANDS.
        fn1 (callable): A function to apply to the power spectral density data for band1.
        band2 (list or str or None): band range of the second band (fmin, fmax) on the denominator. if str, it should be a key in EEG_BANDS. If None, defaults to band1.
        fn2 (callable or None): A function to apply to the power spectral density data for band2. If None, defaults to fn1.

    Returns:
        np.ndarray: The ratio of the statistic in band1 to the statistic in band2.
    
    Usage example for osl preprocess config:
        SLICE_FREQ = 1/0.07  # Hz, frequency for the first slice in init_tracer
        config = {'preproc': [{"init_tracer": {
            'psd_slice': partial(psd_band_ratio, band1=[SLICE_FREQ-1, SLICE_FREQ+1], band2='beta', fn1=np.mean),
            'psd_2slice': partial(psd_band_ratio, band1=[SLICE_FREQ*2-1, SLICE_FREQ*2+1], band2=[20, 35], fn1=np.mean),
            'psd_alpha_maxmed_ratio': partial(psd_band_ratio, band1='alpha', fn1=np.max, fn2=np.median)
        }}]}
    """
    if isinstance(band1, str):
        band1 = EEG_BANDS[band1]
    if isinstance(band2, str):
        band2 = EEG_BANDS[band2]
    if band2 is None:
        band2 = band1
    if fn2 is None:
        fn2 = fn1
    
    if len(band1) != 2 or len(band2) != 2:
        raise ValueError("Both band1 and band2 must be lists of two elements (fmin, fmax).")

    return fn1(psd.get_data(fmin=band1[0], fmax=band1[1]), axis=1) / fn2(psd.get_data(fmin=band2[0], fmax=band2[1]), axis=1)

def psd_band_stat(psd, band, fn=np.mean):
    """Calculate the statistic of the power spectral density in a specified frequency range.

    Args:
        psd (Spectrum): The power spectral density data.
        band (list or str): band range (fmin, fmax) or a key in EEG_BANDS.
        fn (callable): A function to apply to the power spectral density data (default is np.mean, for mean power).

    Returns:
        np.ndarray: The statistic of the mean power in the specified frequency range.

    Not to be confused with :func:`osl_ephys.utils.mean_psd_in_band` (re-exported
    at ``semp.metric.mean_psd_in_band``). They look similar but are not
    interchangeable:

    ============  ================================  ==============================
    aspect        ``psd_band_stat`` (this)          ``mean_psd_in_band``
    ============  ================================  ==============================
    input         an ``mne`` ``Spectrum`` object    a plain numpy PSD row + freqs
    band spec     ``[fmin, fmax]`` / EEG_BANDS key  centre +/- half_width
    reducer       any ``fn`` (mean, kurtosis, ...)  always the mean
    empty band    ``fn`` over an empty slice        falls back to nearest bin
    used by       tracer metrics (init_tracer)      slice_reject harmonic scoring
    ============  ================================  ==============================

    The nearest-bin fallback is why ``slice_reject`` uses ``mean_psd_in_band``
    for its tight slice-harmonic windows; the ``Spectrum``/``fn`` flexibility is
    why the tracer metrics use ``psd_band_stat``.

    Usage example:
        config = {'preproc': [{"init_tracer": {
            'psd_alpha_mean': partial(psd_band_stat, band='alpha', fn=np.mean),
            'psd_kurtosis': partial(psd_band_stat, band=[1,40], fn=kurtosis)
        }}]}
    """
    if isinstance(band, str):
        band = EEG_BANDS[band]
    fmin, fmax = band
    return fn(psd.get_data(fmin=fmin, fmax=fmax), axis=1)

