"""Reporting tool for preprocessing.

"""

# Authors: Andrew Quinn <a.quinn@bham.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Mats van Es <mats.vanes@psych.ox.ac.uk>

import os
import mne
import sys
import yaml
import sails
import argparse
import tempfile
import pickle
import pathlib
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from jinja2 import Template
from tabulate import tabulate
from mne.channels.channels import channel_type
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from pathlib import Path

try:
    from scipy.ndimage import uniform_filter1d
except ImportError:
    from scipy.ndimage.filters import uniform_filter1d

from ..utils import process_file_inputs, validate_outdir
from ..utils.logger import log_or_print
from ..source_recon.parcellation import plot_source_topo


# ----------------------------------------------------------------------------------
# Report generation
mne.viz.set_browser_backend("matplotlib")

def gen_report_from_fif(infiles, outdir, ftype=None, logsdir=None, run_id=None):
    """Generate web-report for a set of MNE data objects.

    Parameters
    ----------
    infiles : list of str
        List of paths to fif files.
    outdir : str
        Directory to save HTML report and figures to.
    ftype : str
        Type of fif file, e.g., ``'raw'`` or ``'preproc-raw'``.
    logsdir : str
        Directory the log files were saved into. If None, log files are assumed
        to be in outdir.replace('report', 'logs')
    run_id : str
        Run ID.
    """
    from ..preprocessing import read_dataset

    # Validate input files and directory to save html file and plots to
    infiles, outnames, good_files = process_file_inputs(infiles)
    outdir = validate_outdir(outdir)

    # Generate HTML data
    for infile in infiles:
        print("Generating report for", infile)
        dataset = read_dataset(infile, ftype=ftype)
        run_id = get_header_id(dataset['raw'])
        htmldatadir = validate_outdir(outdir / run_id)
        gen_html_data(
            dataset['raw'],
            htmldatadir,
            ica=dataset['ica'],
            logsdir=logsdir,
            run_id=run_id,
        )

    # Create report
    gen_html_page(outdir)

    print("************" + "*" * len(str(outdir)))
    print(f"* REPORT: {outdir} *")
    print("************" + "*" * len(str(outdir)))


def get_header_id(raw):
    """Extract scan name from MNE data object.
    
    Parameters
    ----------
    raw : mne.io.:py:class:`mne.io.Raw <mne.io.Raw>`.
        MNE Raw object.
        
    Returns
    -------
    id : str
        Scan name.
    
    """
    return str(raw.filenames[0]).split('/')[-1].strip('.fif')


def gen_html_data(
    raw, outdir, ica=None, preproc_fif_filename=None, logsdir=None, run_id=None
):
    """Generate HTML web-report for an MNE data object.

    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE Raw object.
    outdir : string
        Directory to write HTML data and plots to.
    ica : :py:class:`mne.preprocessing.ICA <mne.preprocessing.ICA>`
        ICA object.
    preproc_fif_filename : str
        Filename of file output by preprocessing
    logsdir : str
        Directory the log files were saved into. If None, log files are assumed
        to be in reportdir.replace('report', 'logs')
    run_id : str
        Run ID.
    """

    data = {}
    data['filename'] = raw.filenames[0]
    data['preproc_fif_filename'] = preproc_fif_filename
    data['fif_id'] = run_id or get_header_id(raw)

    # Scan info
    data['projname'] = raw.info['proj_name']
    data['edataperimenter'] = raw.info['experimenter']
    data['meas_date'] = raw.info['meas_date'].__str__()

    data['acq_samples'] = raw.n_times
    data['acq_sfreq'] = raw.info['sfreq']
    data['acq_duration'] = raw.n_times/raw.info['sfreq']

    # Channels/coils
    data['nchans'] = raw.info['nchan']

    try:
        data['nhpi'] = len(raw.info['hpi_meas'][0]['hpi_coils'])
    except:
        log_or_print("No HPI info in fif file")

    chtype = [channel_type(raw.info, c) for c in range(data['nchans'])]
    chs, chcounts = np.unique(chtype, return_counts=True)
    data['chantable'] = tabulate(np.c_[chs, chcounts], tablefmt='html',
                                 headers=['Channel Type', 'Number Acquired'])

    # Head digitisation
    dig_codes = ('Cardinal', 'HPI', 'EEG', 'Extra')
    dig_counts = np.zeros((4,))

    # Only run this if digitisation is available
    if raw.info['dig']:
        for ii in range(1, 5):
            dig_counts[ii-1] = np.sum([d['kind'] == ii for d in raw.info['dig']])
    #d, dcounts = np.unique(digs, return_counts=True)
    data['digitable'] = tabulate(np.c_[dig_codes, dig_counts], tablefmt='html',
                                 headers=['Digitisation Stage', 'Points Acquired'])

    # Events
    stim_channel = mne.pick_types(raw.info, meg=False, ref_meg=False, stim=True)
    if len(stim_channel) > 0:
        ev = mne.find_events(raw, min_duration=5/raw.info['sfreq'], verbose=False)
        ev, evcounts = np.unique(ev[:, 2], return_counts=True)
        data['eventtable'] = tabulate(np.c_[ev, evcounts], tablefmt='html',
                                      headers=['Event Code', 'Value'])

    # Bad segments
    durs = np.array([r['duration'] for r in raw.annotations])
    full_dur = raw.n_times/raw.info['sfreq']
    types = [r['description'] for r in raw.annotations]

    data['bad_seg'] = []
    
    for modality in ['grad', 'mag', 'eeg']:

        # set bad segs to mean
        bad_tc = np.zeros(raw.n_times)
        for aa in raw.annotations:
            if "bad_segment" in aa["description"] and aa["description"].find(modality) > 0:
                time_inds = np.where((raw.times >= aa["onset"]-raw.first_time) & (raw.times <= (aa["onset"] + aa["duration"] - raw.first_time)))[0]
                bad_tc[time_inds] = 1
        
        mod_dur = np.sum(bad_tc)/raw.info['sfreq']
        pc = np.sum(bad_tc) / len(bad_tc) * 100

        s = 'Modality {0} - {1:.2f}/{2} seconds rejected     ({3:.2f}%)'
        if full_dur > 0:
            data['bad_seg'].append(s.format(modality, mod_dur, full_dur, pc))
            # For summary report:
            data['bad_seg_pc_' + modality] = pc        
                
    # Bad channels
    bad_chans = raw.info['bads']
    if len(bad_chans) == 0:
        data['bad_chans'] = 'No bad channels.'
    else:
        data['bad_chans'] = 'Bad channels: {}'.format(', '.join(bad_chans))
    # For summary report:
    data['bad_chan_num'] = len(bad_chans)

    # Path to save plots
    savebase = str(outdir / '{0}.png')
    
    # Generate plots for the report
    data["plt_config"] = plot_flowchart(raw, savebase)
    data["plt_rawdata"] = plot_rawdata(raw, savebase)
    data['plt_temporalsumsq'] = plot_channel_time_series(raw, savebase, exclude_bads=False)
    data['plt_temporalsumsq_exclude_bads'] = plot_channel_time_series(raw, savebase, exclude_bads=True)
    data['plt_badchans'] = plot_sensors(raw, savebase)
    data['plt_channeldev'] = plot_channel_dists(raw, savebase, exclude_bads=False)
    data['plt_channeldev_exclude_bads'] = plot_channel_dists(raw, savebase, exclude_bads=True)
    data['plt_spectra'], data['plt_zoomspectra'] = plot_spectra(raw, savebase)
    data['plt_freqbands'] = plot_freqbands(raw, savebase)
    data['plt_digitisation'] = plot_digitisation_2d(raw, savebase)
    data['plt_artefacts_eog'] = plot_eog_summary(raw, savebase)
    data['plt_artefacts_ecg'] = plot_ecg_summary(raw, savebase)
    #filenames = plot_artefact_scan(raw, savebase)
    #data.update(filenames)

    # add extra funcs if they exist
    extra_funcs = re.findall(
        "%% extra_funcs start %%(.*?)%% extra_funcs end %%",
        raw.info["description"],
        flags=re.DOTALL,
    )
    if len(extra_funcs) > 0:
        data["extra_funcs"] = ""
        for func in extra_funcs:
            data["extra_funcs"] = data["extra_funcs"] + func
    
    # Add ICA if it's been passed
    if ica is not None:
        data['ica_ncomps_rej'] = len(ica.exclude)
        data['ica_ncomps_rej_ecg'] = [len(ica.labels_['ecg']) if 'ecg' in ica.labels_ else 'N/A'][0]
        data['ica_ncomps_rej_eog'] = [len(ica.labels_['eog']) if 'eog' in ica.labels_ else 'N/A'][0]
        if len(ica.exclude) > 0:
            data['plt_ica'] = plot_bad_ica(raw, ica, savebase)

    # add maxfilter info if possible
    # we have to guess the exact filename
    g=[]
    for ext in ['tsss', 'trans', 'transdef', 'autobad', 'autobad_sss']:
        g.append(glob(str(data['filename']).replace(f"{ext}.fif",'') +  '*.log'))
    g = list(np.concatenate(g))
    
    for ig in g:
        with open(ig, 'r') as log_file:
            if 'autobad' in ig:
                data['maxlog_autobad'] = log_file.read()
            elif 'sss' in ig:
                data['maxlog_sss'] = log_file.read()
            elif 'trans' in ig:
                data['maxlog_trans'] = log_file.read()
    
    # add log files if possible
    if logsdir is None:
        logsdir = outdir._str.replace('preproc_report', 'logs')
    elif type(logsdir)==pathlib.PosixPath:
        logsdir = os.path.join(logsdir._str, outdir._str.split('/')[-1])
        
    if os.path.exists(logsdir + '.log'):
        with open(logsdir + '.log', 'r') as log_file:
            data['log'] = log_file.read()
        
    if os.path.exists(logsdir + '.error.log'):
        with open(logsdir + '.error.log', 'r') as log_file:
            data['errlog'] = log_file.read()

    # Save data that will be used to create html page
    with open(outdir / 'data.pkl', 'wb') as outfile:
        pickle.dump(data, outfile)


def gen_html_page(outdir, logsdir=None):
    """Generate an HTML page from a report directory.

    Parameters
    ----------
    outdir : str
        Directory to generate HTML report with.
        
    Returns
    -------
    success : bool
        Whether the report was successfully generated.
    """
    outdir = pathlib.Path(outdir)

    # Subdirectories which contains plots for each fif file
    subdirs = sorted(
        [d.stem for d in pathlib.Path(outdir).iterdir() if d.is_dir()]
    )

    # Load data for each fif file
    data = []
    for subdir in subdirs:
        subdir = pathlib.Path(subdir)
        # Just generate the html page with the successful runs
        try:
            data.append(pickle.load(open(outdir / subdir / "data.pkl", "rb")))
        except:
            pass

    total = len(data)
    if total == 0:
        return False

    # Add info to data indicating the total number of files
    # and an id for each file
    for i in range(total):
        data[i]["num"] = i + 1
        data[i]["total"] = total

    # Create panels
    panels = []
    panel_template = load_template('raw_subject_panel')
    for i in range(total):
        panels.append(panel_template.render(data=data[i]))

    # Hyperlink to each panel on the page
    filenames = ""
    for i in range(total):
        filename = pathlib.Path(data[i]["preproc_fif_filename"]).name.replace('_preproc-raw.fif', '')
        filenames += "{0}. {1}<br>".format(i + 1, filename)

    # Render the full page
    page_template = load_template('subject_report')
    page = page_template.render(panels=panels, filenames=filenames)

    # Write the output file
    outpath = pathlib.Path(outdir) / 'subject_report.html'
    with open(outpath, 'w') as f:
        f.write(page)

    return True


def gen_html_summary(reportdir, logsdir=None):
    """Generate an HTML summary from a report directory.

    Parameters
    ----------
    reportdir : str
        Directory to generate HTML summary report with.
    logsdir : str
        Directory the log files were saved into. If None, log files are assumed
        to be in reportdir.replace('report', 'logs')
    """
    reportdir = Path(reportdir)

    # Subdirectories which contains plots for each fif file
    subdirs = sorted(
        [d.stem for d in Path(reportdir).iterdir() if d.is_dir()]
    )

    # Load HTML data
    subject_data = []
    for subdir in subdirs:
        subdir = Path(subdir)
        # Just generate the html page with the successful runs
        try:
            subject_data.append(
                pickle.load(open(reportdir / subdir / "data.pkl", "rb"))
            )
        except:
            pass

    total = len(subject_data)
    if total == 0:
        return False

    # Data used in the summary report
    data = {}
    data["total"] = total

    # Create plots
    os.makedirs(f"{reportdir}/summary", exist_ok=True)

    data["plt_config"] = subject_data[0]["plt_config"]
    if "extra_funcs" in subject_data[0]:
        data["extra_funcs"] = subject_data[0]["extra_funcs"]
    data['tbl'] = gen_summary_data(subject_data)
    
    # log files
    if logsdir is None:
        logsdir = reportdir.__str__().replace('preproc_report', 'logs')
    elif type(logsdir)==pathlib.PosixPath:
        logsdir = logsdir.__str__()
        
    if os.path.exists(os.path.join(logsdir, 'batch_preproc.log')):
        with open(os.path.join(logsdir, 'batch_preproc.log'), 'r') as log_file:
            data['batchlog'] = log_file.read()
    
    g = glob(os.path.join(logsdir, '*.error.log'))    
    if len(g)>0:
        data['errlog'] = {}
        for ig in sorted(g):
            with open(ig, 'r') as log_file:
                data['errlog'][ig.split('/')[-1].split('.error.log')[0]] = log_file.read()
            
    # Create panel
    panel_template = load_template('raw_summary_panel')
    panel = panel_template.render(data=data)

    # List of filenames
    filenames = ""
    for i in range(total):
        filename = Path(subject_data[i]["filename"]).name
        filenames += "{0}. {1}<br>".format(i + 1, filename)

    # Render the full page
    page_template = load_template('summary_report')
    page = page_template.render(panel=panel, filenames=filenames)

    # Write the output file
    outpath = Path(reportdir) / 'summary_report.html'
    with open(outpath, 'w') as f:
        f.write(page)

    return True


def gen_summary_data(subject_data):
    df = pd.DataFrame()
    column_headers = ['Session ID', 'Measurement month', 'Duration (s)', 'Bad segments (%) - mag', 
                      'Bad segments (%) - grad)', 'Bad segments (%) - eeg)', 'Bad channels (#)', 
                      'Bad channels (id)', 'Bad ICA (total)', 'Bad ICA (ECG)', 'Bad ICA (EOG)']
    fields = ['fif_id', 'meas_date', 'acq_duration', 'bad_seg_pc_mag', 'bad_seg_pc_grad', 
              'bad_seg_pc_eeg', 'bad_chan_num', 'bad_chans', 'ica_ncomps_rej', 
              'ica_ncomps_rej_ecg', 'ica_ncomps_rej_eog']
    for field, hdr in zip(fields, column_headers):
        if field == 'meas_date': # anonymise date (only year/month)
            df[hdr] = ["-".join(subject_data[i][field].split(' ')[0].split('-')[:-1]) if field in subject_data[i].keys() else None for i in range(len(subject_data))]
        else:
            df[hdr] = [subject_data[i][field] if field in subject_data[i].keys() else None for i in range(len(subject_data))]
    
    # remove columns that only contains None
    df = df.dropna(axis=1, how='all')
    df.index += 1 # start index at 1
    tbl = df.to_html(classes="display", table_id="preproc_tbl")
    
    return tbl


def load_template(tname):
    """Load an HTML template from the templates directory.

    Parameters
    ----------
    tname : str
        Name of the template to load.
        
    Returns
    -------
    template : jinja2.Template
        The loaded template.
    """
    basedir = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(basedir, 'templates', '{0}.html'.format(tname))
    with open(fname, 'r') as file:
        template = Template(file.read())
    return template


# ----------------------------------------------------------------------------------
# Scan stats and figures

def plot_flowchart(raw, savebase=None):
    """Plots preprocessing flowchart(s)
    
    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE Raw object.
    savebase : str
        Base string for saving figures.
        
    Returns
    -------
    fpath : str
        Path to saved figure.    
    
    """
    from ..preprocessing import get_config_from_fif, plot_preproc_flowchart
    # Get config info from raw.info['description']
    config_list = get_config_from_fif(raw)

    # Number of subplots, i.e. the number of times osl-ephys preprocessing was applied
    nrows = len(config_list)

    if nrows == 0:
        # No preprocessing was applied
        return None

    # Plot flowchart in subplots
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 6*(nrows+1)))

    cnt=0
    for config in config_list:
        if type(ax)==np.ndarray:
            axes=ax[cnt]
            title=f"osl-ephys Preprocessing Stage {cnt+1}"
        else:
            axes=ax
            title=None
        fig, axes = plot_preproc_flowchart(config, fig=fig, ax=axes, title=title)
        cnt=cnt+1
    if savebase is not None:
        # Save figure
        figname = savebase.format('flowchart')
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

        # Return the filename
        savebase = pathlib.Path(savebase)
        filebase = savebase.parent.name + "/" + savebase.name
        fpath = filebase.format('flowchart')
    else:
        fpath = None
    return fpath    
        

def plot_rawdata(raw, savebase):
    """Plots raw data.
    
    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE Raw object.
    savebase : str
        Base string for saving figures.
        
    Returns
    -------
    fpath : str
        Path to saved figure.      
    
    """

    fig = raw.pick(['meg', 'eeg']).plot(n_channels=np.inf, duration=raw.times[-1], use_opengl=False, show=False)

    if savebase is not None:
        figname = savebase.format('rawdata')
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

        # Return the filename
        savebase = pathlib.Path(savebase)
        filebase = savebase.parent.name + "/" + savebase.name
        fpath = filebase.format('rawdata')
    else:
        fpath = None
    return fpath


def plot_channel_time_series(raw, savebase=None, exclude_bads=False):
    """Plots sum-square time courses.
    
    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE Raw object.
    savebase : str
        Base string for saving figures.
    exclude_bads : bool
        Whether to exclude bad channels and bad segments.
        
    Returns
    -------
    fpath : str
        Path to saved figure.    
    
    """

    if exclude_bads:
        # excludes bad channels and bad segments
        exclude = 'bads'
    else:
        # includes bad channels and bad segments
        exclude = []

    is_ctf = raw.info["dev_ctf_t"] is not None

    if is_ctf:

        # Note that with CTF mne.pick_types will return:
        # ~274 axial grads (as magnetometers) if {picks: 'mag', ref_meg: False}
        # ~28 reference axial grads if {picks: 'grad'}

        channel_types = {
            'Axial Grads (chtype=mag)': mne.pick_types(
                raw.info, meg='mag', ref_meg=False, exclude=exclude
            ),
            'Ref Axial Grad (chtype=ref_meg)': mne.pick_types(
                raw.info, meg='grad', exclude=exclude
            ),
            'EEG': mne.pick_types(raw.info, eeg=True),
            'CSD': mne.pick_types(raw.info, csd=True),
        }
    else:
        channel_types = {
            'Magnetometers': mne.pick_types(raw.info, meg='mag', exclude=exclude),
            'Gradiometers': mne.pick_types(raw.info, meg='grad', exclude=exclude),
            'EEG': mne.pick_types(raw.info, eeg=True),
            'CSD': mne.pick_types(raw.info, csd=True),
        }

    t = raw.times
    x = raw.get_data()

    # Number of subplots, i.e. the number of different channel types in the fif file
    nrows = 0
    for _, c in channel_types.items():
        if len(c) > 0:
            nrows += 1

    if nrows == 0:
        return None

    # Make sum-square plots
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(16, 4))
    if nrows == 1:
        ax = [ax]
    row = 0
    for name, chan_inds in channel_types.items():
        if len(chan_inds) == 0:
            continue
        ss = np.sum(x[chan_inds] ** 2, axis=0)

        # calculate ss value to give to bad segments for plotting purposes
        good_data = raw.get_data(picks=chan_inds, reject_by_annotation='NaN')
        # get indices of good data
        good_inds = np.where(~np.isnan(good_data[0,:]))[0]
        ss_bad_value = np.mean(ss[good_inds])

        if exclude_bads:
            # set bad segs to mean
            for aa in raw.annotations:
                if "bad_segment" in aa["description"]:
                    time_inds = np.where((raw.times >= aa["onset"]-raw.first_time) & (raw.times <= (aa["onset"] + aa["duration"] - raw.first_time)))[0]
                    ss[time_inds] = ss_bad_value

        ss = uniform_filter1d(ss, int(raw.info['sfreq']))

        ax[row].plot(t, ss)
        ax[row].legend([name], frameon=False, fontsize=16)
        ax[row].set_xlim(t[0], t[-1])
        ylim = ax[row].get_ylim()
        for a in raw.annotations:
            if "bad_segment" in a["description"]:
                ax[row].axvspan(
                    a["onset"] - raw.first_time,
                    a["onset"] + a["duration"] - raw.first_time,
                    color="red",
                    alpha=0.8,
                )
        row += 1
    ax[0].set_title('Sum-Square Across Channels')
    ax[-1].set_xlabel('Time (seconds)')

    # Save
    if savebase is not None:
        plt.tight_layout()
        if exclude_bads:
            plot_name = 'temporal_sumsq_exclude_bads'
        else:
            plot_name = 'temporal_sumsq'
        figname = savebase.format(plot_name)
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

        # Return the filename
        savebase = pathlib.Path(savebase)
        filebase = savebase.parent.name + "/" + savebase.name
        fpath = filebase.format(plot_name)
    else:
        fpath = None
    return fpath


def plot_sensors(raw, savebase=None):
    """Plots sensors with bad channels highlighted.
    
    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE Raw object.
    savebase : str
        Base string for saving figures.
        
    Returns
    -------
    fpath : str
        Path to saved figure.    
    
    """
    # plot channel types seperately for neuromag306 (3 coils in same location)

    if 3012 in np.unique([i['coil_type'] for i in raw.info['chs']]):
        with open(str(Path(__file__).parent.parent) + "/utils/neuromag306_info.yml", 'r') as f:
            channels = yaml.safe_load(f)
        if 3024 in np.unique([i['coil_type'] for i in raw.info['chs']]):
            coil_types = ['mag', 'grad_longitude', 'grad_lattitude']
        else:
            coil_types = ['grad_longitude', 'grad_lattitude']
        fig, ax = plt.subplots(1,len(coil_types), figsize=(16,4))
        for k in range(len(coil_types)):
            raw.copy().pick_channels(channels[coil_types[k]]).plot_sensors(axes=ax[k], show=False)
            ax[k].set_title(f"{coil_types[k].replace('_', ' ')}")
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        ax[0].axis('off')
        ax[2].axis('off')
        raw.plot_sensors(show=False, axes=ax[1])

    if savebase is not None:
        figname = savebase.format('bad_chans')
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

        # Return the filename
        savebase = pathlib.Path(savebase)
        filebase = savebase.parent.name + "/" + savebase.name
        fpath = filebase.format('bad_chans')
    else:
        fpath = None
    return fpath


def plot_channel_dists(raw, savebase=None, exclude_bads=True):
    """Plot distributions of temporal standard deviation.
    
    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE Raw object.
    savebase : str 
        Base string for saving figures.
    exclude_bads : bool
        Whether to exclude bad channels and bad segments.
        
    Returns
    -------
    fpath : str
        Path to saved figure.    
    """

    if exclude_bads:
        # excludes bad channels and bad segments
        exclude = 'bads'
        reject_by_annotation = 'omit'
    else:
        # includes bad channels and bad segments
        exclude = []
        reject_by_annotation = None

    is_ctf = raw.info["dev_ctf_t"] is not None

    if is_ctf:

        # Note that with CTF mne.pick_types will return:
        # ~274 axial grads (as magnetometers) if {picks: 'mag', ref_meg: False}
        # ~28 reference axial grads if {picks: 'grad'}
        channel_types = {
            'Axial Grads (chtype=''mag'')': mne.pick_types(raw.info, meg='mag', ref_meg=False, exclude=exclude),
            'Ref Axial Grad (chtype=''ref_meg'')': mne.pick_types(raw.info, meg='grad', exclude=exclude),
            'EEG': mne.pick_types(raw.info, eeg=True, exclude=exclude),
            'CSD': mne.pick_types(raw.info, csd=True, exclude=exclude),
        }
    else:
        channel_types = {
            'Magnetometers': mne.pick_types(raw.info, meg='mag', exclude=exclude),
            'Gradiometers': mne.pick_types(raw.info, meg='grad', exclude=exclude),
            'EEG': mne.pick_types(raw.info, eeg=True, exclude=exclude),
            'CSD': mne.pick_types(raw.info, csd=True, exclude=exclude),
        }

    x = raw.get_data(reject_by_annotation=reject_by_annotation)

    # Number of subplots, i.e. the number of different channel types in the fif file
    ncols = 0
    for _, c in channel_types.items():
        if len(c) > 0:
            ncols += 1

    if ncols == 0:
        return None
    
    # Make plots
    fig, ax = plt.subplots(nrows=1, ncols=ncols, figsize=(9, 3.5))
    if ncols == 1:
        ax = [ax]
    row = 0
    for name, chan_inds in channel_types.items():
        if len(chan_inds) == 0:
            continue

        ax[row].hist(x[chan_inds, :].std(axis=1), bins=24, histtype='step')
        ax[row].legend(['Temporal Std-Dev'], frameon=False)
        ax[row].set_xlabel('Std-Dev')
        ax[row].set_ylabel('Channel Count')
        ax[row].set_title(name)
        row += 1

    if exclude_bads:
        # excludes bad channels and bad segments
        plot_name = 'channel_dev_exclude_bads'
    else:
        # includes bad channels and bad segments
        plot_name = 'channel_dev'

    # Save
    if savebase is not None:
        plt.tight_layout()
        figname = savebase.format(plot_name)
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)
        # Return the filename
        savebase = pathlib.Path(savebase)
        filebase = savebase.parent.name + "/" + savebase.name
        fpath = filebase.format(plot_name)
    else:
        fpath = None
    return fpath


def plot_spectra(raw, savebase=None):
    """Plot power spectra for each sensor modality.
    
    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`   
        MNE Raw object.
    savebase : str  
        Base string for saving figures.
        
    Returns
    -------
    fpath1 : str
        Path to saved figure (full spectra).
    fpath2 : str
        Path to saved figure (zoomed in spectra).
    """

    is_ctf = raw.info["dev_ctf_t"] is not None
    if is_ctf:
        # Note that with CTF mne.pick_types will return:
        # ~274 axial grads (as magnetometers) if {picks: 'mag', ref_meg: False}
        # ~28 reference axial grads if {picks: 'grad'}
        channel_types = {
            'Axial Grad (chtype=''mag'')': mne.pick_types(raw.info, meg='mag', ref_meg=False, exclude='bads'),
            'EEG': mne.pick_types(raw.info, eeg=True, exclude='bads'),
        }
    else:
        channel_types = {
            'Magnetometers': mne.pick_types(raw.info, meg='mag', exclude='bads'),
            'Gradiometers': mne.pick_types(raw.info, meg='grad', exclude='bads'),
            'EEG': mne.pick_types(raw.info, eeg=True, exclude='bads'),
        }

    # Number of subplots, i.e. the number of different channel types in the fif file
    nrows = 0
    for _, c in sorted(channel_types.items()):
        if len(c) > 0:
            nrows += 1

    if nrows == 0:
        return None

    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(8, 7))
    if nrows == 1:
        ax = [ax]
    row = 0

    for name, chan_inds in sorted(channel_types.items()):
        if len(chan_inds) == 0:
            continue

        # Plot spectra
        raw.compute_psd(
            picks=chan_inds, 
            n_fft=int(raw.info['sfreq']*2),
            verbose=0).plot(
                        axes=ax[row],
                        show=False)

        ax[row].set_title(name, fontsize=12)

        row += 1

    # Save full spectra
    if savebase is not None:
        figname = savebase.format('spectra_full')
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

    # Make plots
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(8, 7))
    if nrows == 1:
        ax = [ax]
    row = 0

    for name, chan_inds in sorted(channel_types.items()):
        if len(chan_inds) == 0:
            continue

        # Plot zoomed in spectra
        raw.compute_psd(
        picks=chan_inds, 
        fmin=1,
        fmax=48,
        n_fft=int(raw.info['sfreq']*2),
        verbose=0).plot(
                    axes=ax[row],
                    show=False)

        ax[row].set_title(name, fontsize=12)

        row += 1

    # Save zoomed in spectra
    if savebase is not None:
        figname = savebase.format('spectra_zoom')
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

        # Return filenames
        savebase = pathlib.Path(savebase)
        filebase = savebase.parent.name + "/" + savebase.name
        fpath1 = filebase.format('spectra_full')
        fpath2 = filebase.format('spectra_zoom')
    else:
        fpath1 = None
        fpath2 = None
    return fpath1, fpath2


def plot_freqbands(raw, savebase=None):
    """Plot power spectra for each sensor modality.
    
    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`   
        MNE Raw object.
    savebase : str  
        Base string for saving figures.
        
    Returns
    -------
    figname : str
        Path to saved figure.
    """
    print(f"TYPE: {type(raw)}")
    print(f"INFO: {raw.info}")
    if 'dev_ctf_t' in raw.info.keys() and raw.info["dev_ctf_t"] is not None:
        is_ctf = True
    else:
        is_ctf = False
        
    is_parc = np.any(['parcel' in ch for ch in raw.info['ch_names']])
    if is_ctf:
        # Note that with CTF mne.pick_types will return:
        # ~274 axial grads (as magnetometers) if {picks: 'mag', ref_meg: False}
        # ~28 reference axial grads if {picks: 'grad'}
        channel_types = {
            'Axial Grad (chtype=''mag'')': mne.pick_types(raw.info, meg='mag', ref_meg=False, exclude='bads'),
            'EEG': mne.pick_types(raw.info, eeg=True, exclude='bads'),
        }
    elif is_parc:
        channel_types = {
            'Parcel': mne.pick_types(raw.info, misc=True, ref_meg=False, exclude='bads'),
        }
    else:
        channel_types = {
            'Magnetometers': mne.pick_types(raw.info, meg='mag', exclude='bads'),
            'Gradiometers': mne.pick_types(raw.info, meg='grad', exclude='bads'),
            'EEG': mne.pick_types(raw.info, eeg=True, exclude='bads'),
        }

    # Number of subplots, i.e. the number of different channel types in the fif file
    nrows = 0
    for _, c in sorted(channel_types.items()):
        if len(c) > 0:
            nrows += 1

    if nrows == 0:
        return None
    
    freq_bands = [(1,4), (4,8), (8,13), (13,30), (30,80), (80,150)]
    ix = [i for i in range(len(freq_bands)) if np.logical_and(freq_bands[i][1] > raw.info['highpass'], freq_bands[i][0] < raw.info['lowpass'])]
    freq_bands = [freq_bands[i] for i in ix]
    freq_band_names = [['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'High Gamma'][i] for i in ix]
    ncols = len(freq_bands)

    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 0.5 + 1.5*nrows))
    
    row = 0
    for name, chan_inds in sorted(channel_types.items()):
        if len(chan_inds) == 0:
            continue

        # Plot spectra
        psd = raw.compute_psd(
            picks=chan_inds, 
            n_fft=int(raw.info['sfreq']*2),
            verbose=0)
        
        if nrows==1:
            iax = ax
        else:
            iax = ax[row]
        
        for ifrq, (f1, f2) in enumerate(freq_bands):
            inds = np.logical_and(psd.freqs>f1, psd.freqs<=(f2))
            p = psd.data[:, inds].mean(axis=1)
            if is_parc: # normalize because nilearn doesn't plot small values
                plot_source_topo(p/p.std(), axis=iax[ifrq], cmap='Reds') 
            else:
                mne.viz.plot_topomap(p, psd.info, axes=iax[ifrq])
            if row==0:
                iax[ifrq].set_title(f"{freq_band_names[ifrq]}\n({f1}-{f2} Hz)")
            if ifrq==0:
                iax[ifrq].text(-0.1, 0.5, name, transform=iax[ifrq].transAxes, 
                               rotation=90, va='center', ha='center', fontsize=10)

        row += 1

    # Save power bands spectra
    if savebase is not None:
        figname = savebase.format('freqbands')
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)
        return figname
    else:
        return None


def plot_digitisation_2d(raw, savebase=None):
    """Plots the digitisation and headshape.
    
    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE Raw object.
    savebase : str
        Base string for saving figures.
        
    Returns
    -------
    fpath : str
        Path to saved figure.
    """

    # Return if no digitisation available
    if not raw.info['dig']:
        return None

    # Make plot
    fig = plt.figure(figsize=(16, 4))

    plt.subplot(141)
    plt.gca().set_aspect('equal')
    for dp in raw.info['dig']:
        if dp['kind'] == 1:
            plt.plot(dp['r'][0], dp['r'][1], 'r^')
        if dp['kind'] == 2:
            plt.plot(dp['r'][0], dp['r'][1], 'mo')
        if dp['kind'] == 3:
            plt.plot(dp['r'][0], dp['r'][1], 'g+')
        if dp['kind'] == 4:
            plt.plot(dp['r'][0], dp['r'][1], 'b.')
    plt.title('Top View')

    plt.subplot(142)
    plt.gca().set_aspect('equal')
    for dp in raw.info['dig']:
        if dp['kind'] == 1:
            plt.plot(dp['r'][0], dp['r'][2], 'r^')
        if dp['kind'] == 2:
            plt.plot(dp['r'][0], dp['r'][2], 'mo')
        if dp['kind'] == 3:
            plt.plot(dp['r'][0], dp['r'][2], 'g+')
        if dp['kind'] == 4:
            plt.plot(dp['r'][0], dp['r'][2], 'b.')
    plt.title('Front View')

    plt.subplot(143)
    plt.gca().set_aspect('equal')
    for dp in raw.info['dig']:
        if dp['kind'] == 1:
            plt.plot(dp['r'][1], dp['r'][2], 'r^')
        if dp['kind'] == 2:
            plt.plot(dp['r'][1], dp['r'][2], 'mo')
        if dp['kind'] == 3:
            plt.plot(dp['r'][1], dp['r'][2], 'g*')
        if dp['kind'] == 4:
            plt.plot(dp['r'][1], dp['r'][2], 'b.')
    plt.title('Side View')

    plt.subplot(144, frameon=False)
    plt.xticks([])
    plt.yticks([])
    legend_elements = [Line2D([0], [0], marker='^', color='w', lw=4, label='Fiducial', markerfacecolor='r',markersize=14),
                       Line2D([0], [0], marker='o', color='w', label='HPI', markerfacecolor='m', markersize=14),
                       Line2D([0], [0], marker='*', color='w', label='EEG', markerfacecolor='g', markersize=14),
                       Line2D([0], [0], marker='.', color='w', label='Headshape', markerfacecolor='b', markersize=14)]
    plt.legend(handles=legend_elements, loc='center', frameon=False)

    if savebase is not None:
        plt.tight_layout()
        figname = savebase.format('digitisation')
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

        # Return the filename
        savebase = pathlib.Path(savebase)
        filebase = savebase.parent.name + "/" + savebase.name
        fpath = filebase.format('digitisation')
    else:
        fpath = None
    return fpath


def plot_headmovement(raw, savebase=None):
    """Plot headmovement - WORK IN PROGRESS... seems v-slow atm"""
    chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
    chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=False)
    fig = mne.viz.plot_head_positions(head_pos, mode='traces')
    if savebase is not None:
        figname = savebase.format('headpos')
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)


def plot_eog_summary(raw, savebase=None):
    """Plot raw EOG time series."""

    # Get the raw EOG data
    chan_inds = mne.pick_types(raw.info, eog=True)
    if len(chan_inds) == 0:
        return None
    t = raw.times
    x = raw.get_data(chan_inds).T

    # Make the plot
    fig, ax = plt.subplots(figsize=(16, 2))
    ax.plot(t, x)
    ax.set_xlim([t[0], t[-1]])

    # Save
    if savebase is not None:
        plt.tight_layout()
        figname = savebase.format('EOG')
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

        # Return the filename
        savebase = pathlib.Path(savebase)
        filebase = savebase.parent.name + "/" + savebase.name
        fpath = filebase.format('EOG')
    else:
        fpath = None
    return fpath


def plot_ecg_summary(raw, savebase=None):
    """Plot ECG summary.
    
    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE Raw object.
    savebase : str
        Base string for saving figures.
        
    Returns
    -------
    fpath : str
        Path to saved figure.
    """

    # Get the raw ECG data
    chan_inds = mne.pick_types(raw.info, ecg=True)
    if len(chan_inds) == 0:
        return None
    t = raw.times
    x = raw.get_data(chan_inds).T

    # Make the plot
    fig, ax = plt.subplots(figsize=(16, 2))
    ax.plot(t, x)
    ax.set_xlim([t[0], t[-1]])

    # Save
    if savebase is not None:
        plt.tight_layout()
        figname = savebase.format('ECG')
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

        # Return the filename
        savebase = pathlib.Path(savebase)
        filebase = savebase.parent.name + "/" + savebase.name
        fpath = filebase.format('ECG')
    else:
        fpath = None
    return fpath


def plot_bad_ica(raw, ica, savebase):
    """Plot ICA characteristics for rejected components.
    
    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE Raw object.
    ica : :py:class:`mne.preprocessing.ICA <mne.preprocessing.ICA>` 
        MNE ICA object.
    savebase : str  
        Base string for saving figures.
        
    Returns
    -------
    fpath : str
        Path to saved figure.
    
    """

    exclude_uniq = np.sort(np.unique(ica.exclude))[::-1]
    nbad = len(exclude_uniq)

    # Handle the case when ica.labels_ and ica.exclude don't match
    if len(ica.labels_) != nbad:
        # Make a dummy labels_ dict
        ica.labels_ = {ica._ica_names[exc]: '' for exc in exclude_uniq}

    # Create figure
    fig = plt.figure(figsize=(16, 5 * np.max([nbad,1])), facecolor=[0.95] * 3)
    if nbad == 0:
        plt.subplot(111, frameon=False)
        plt.text(0.5, 0.5, 'No Components Rejected', ha='center', va='center')
        plt.xticks([])
        plt.yticks([])
    else:
        axes = []
        for i in np.arange(nbad):
            lowerlimit = 0.1 + i / (nbad * 1.1)
            multiplier = nbad * 1.3

            # Create axis for subplot
            # adapted from mne/viz/ica._create_properties_layout
            if 'mag' in raw.get_channel_types() and 'grad' in raw.get_channel_types():
                topomap1_layout = [0.08, lowerlimit + 0.5 / multiplier, (0.3-0.08)/2+0.08, 0.45 / multiplier]
                topomap2_layout = [(0.3-0.08)/2+0.08, lowerlimit + 0.5 / multiplier, 0.3, 0.45 / multiplier]
            else:
                topomap1_layout = [0.08, lowerlimit + 0.5 / multiplier, 0.3, 0.45 / multiplier]
            axes_params = (('topomap', topomap1_layout),
                           ('image', [0.5, lowerlimit + 0.6 / multiplier, 0.45, 0.35 / multiplier]),
                           ('erp', [0.5, lowerlimit + 0.5 / multiplier, 0.45, 0.1 / multiplier]),
                           ('spectrum', [0.08, lowerlimit + 0.1 / multiplier, 0.32, 0.3 / multiplier]),
                           ('variance', [0.5, lowerlimit + 0.025 / multiplier, 0.45, 0.25 / multiplier]))
            axes += [[fig.add_axes(loc, label=name) for name, loc in axes_params]]

            ica.plot_properties(raw, picks=exclude_uniq[i], axes=axes[i], show=False, verbose=0)

            # Add another topo if we're dealing with two sensor types
            if 'mag' in raw.get_channel_types() and 'grad' in raw.get_channel_types():
                ax2 = fig.add_axes(topomap2_layout, label='topomap2')
                kind, dropped_indices, epochs_src, data = mne.viz.ica._prepare_data_ica_properties(
                    raw, ica, reject_by_annotation=True, reject='auto')
                mne.viz.ica._plot_ica_topomap(ica, exclude_uniq[i], ch_type='grad',  axes=ax2, show=False)

            if np.any([x in ica.labels_.keys() for x in ica._ica_names]): # this is for the osl_plot_ica convention
                title = "".join((ica._ica_names[exclude_uniq[i]]," - ", ica.labels_[ica._ica_names[exclude_uniq[i]]].upper()))

            elif 'eog' in ica.labels_.keys() and 'ecg' in ica.labels_.keys(): # this is for the MNE automatic labelling convention
                flag_eog = exclude_uniq[i] in ica.labels_['eog']
                flag_ecg = exclude_uniq[i] in ica.labels_['ecg']
                title = "".join((ica._ica_names[exclude_uniq[i]]," - ", flag_eog*'EOG', flag_ecg*flag_eog*'/', flag_ecg*'ECG'))

            elif 'eog' in ica.labels_.keys():
                flag_eog = exclude_uniq[i] in ica.labels_['eog']
                title = "".join((ica._ica_names[exclude_uniq[i]]," - ", flag_eog*'EOG'))

            elif 'ecg' in ica.labels_.keys():
                flag_ecg = exclude_uniq[i] in ica.labels_['ecg']
                title = "".join((ica._ica_names[exclude_uniq[i]]," - ", flag_ecg*'ECG'))

            if 'mag' in raw.get_channel_types() and 'grad' in raw.get_channel_types():
                title = title + '\n (mag)'
                ax2.set_title(title.replace('mag', 'grad'))

            else: # this is for if there is nothing in ica.labels_
                title = None

            if title is not None:
                axes[i][0].set_title(title, fontsize=12)
                
            # check if the bads were excluded
            if np.isclose(ica._transform_raw(raw, 0, len(raw.times))[exclude_uniq[i],:].mean(), 0, atol=1e-9):
                fig.text(0.5, lowerlimit + 1 / multiplier, 'Excluded from preprocessed data', ha='center', va='center', fontsize=14, color='black')
            else:
                fig.text(0.5, lowerlimit + 1 / multiplier, 'Not yet excluded from data', ha='center', va='center', fontsize=14, color='red')

    if savebase is not None:
        figname = savebase.format('ica')
        fig.savefig(figname, dpi=150, transparent=True)
        plt.close(fig)

        # Return the filename
        savebase = pathlib.Path(savebase)
        filebase = savebase.parent.name + "/" + savebase.name
        fpath = filebase.format('ica')
    else:
        fpath = None
    return fpath


def plot_summary_bad_segs(subject_data, reportdir):
    """Plot summary of bad channels over subjects.

    Parameters
    ----------
    subject_data : list
        list of data for each subject, typically generated by gen_html_data
    reportdir : str
        Path to report directory. We will save the plot in this directory.


    Returns
    -------
    path : str
        Path to plot.
    """

    output_file = reportdir / "summary/bad_segs.png"
    fig, ax = plt.subplots()

    for modality in ['grad', 'mag', 'eeg']:

        if 'bad_seg_pc_' + modality in subject_data[0].keys():
            pc_segs = []
            for data in subject_data:

                # check if key exists data
                if 'bad_seg_pc_' + modality in data.keys():
                    pc = data['bad_seg_pc_' + modality]
                    pc_segs.append(pc)
                else:
                    print('Session {} missing bad_seg_pc_{}'.format(data['filename'], modality))
                    pc_segs.append(0)

            ax.plot(
                range(1, len(subject_data) + 1),
                pc_segs,
                label=modality,
            )

    ax.legend()
    ax.set_xlabel("Subject")
    ax.set_ylabel("Percentage time marked bad")
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return f"summary/bad_segs.png"


def plot_summary_bad_chans(subject_data, reportdir):
    """Plot summary of bad channels over subjects.

    Parameters
    ----------
    subject_data : list
        list of data for each subject, typically generated by gen_html_data
    reportdir : str
        Path to report directory. We will save the plot in this directory.

    Returns
    -------
    path : str
        Path to plot.
    """

    output_file = reportdir / "summary/bad_chans.png"
    fig, ax = plt.subplots()

    # For summary report:
    num_bads = []
    for data in subject_data:
        num_bads.append(data['bad_chan_num'])

    ax.plot(
        range(1, len(num_bads) + 1),
        num_bads,
    )
    ax.set_xlabel("Subject")
    ax.set_ylabel("Number of bad channels")
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return f"summary/bad_chans.png"

def plot_artefact_scan(raw, savebase=None):
    """Plot artefact scan.
    
    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE Raw object.
    savebase : str
        Base string for saving figures.
        
    Returns
    -------
    fpath : str
        Path to saved figure.
    """
    events = mne.find_events(raw, min_duration=2/raw.info['sfreq'])
    modalities = ['mag', 'grad']

    # Plot eye movements
    event_dict = {'moves': 1}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-1, tmax=16.5,
                        preload=True)
    x = epochs.get_data()

    yl = np.abs(x[:,:2,:]).max() * 1.1

    plt.figure(figsize=(16,5))
    ax = plt.subplot(121)
    patches = [Rectangle((-1,-yl),1,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((0,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((2,-yl),2,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((4,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((6,-yl),2,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((8,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((10,-yl),2,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((12,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((14,-yl),2,yl*2, alpha=0.2, facecolor='grey')]
    for rec in patches:
        ax.add_artist(rec)
    plt.text(-0.5, yl, 'fix', ha='center')
    plt.text(1, yl, 'Up', ha='center')
    plt.text(5, yl, 'Right', ha='center')
    plt.text(9, yl, 'Down', ha='center')
    plt.text(13, yl, 'Left', ha='center')
    plt.plot(epochs.times, x[:,0,:].T)
    plt.ylim(-yl, yl*1.1)
    ax = plt.subplot(122)
    patches = [Rectangle((-1,-yl),1,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((0,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((2,-yl),2,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((4,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((6,-yl),2,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((8,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((10,-yl),2,yl*2, alpha=0.2, facecolor='grey'),
               Rectangle((12,-yl),2,yl*2, alpha=0.2, facecolor='r'),
               Rectangle((14,-yl),2,yl*2, alpha=0.2, facecolor='grey')]
    for rec in patches:
        ax.add_artist(rec)
    plt.text(-0.5, yl, 'fix', ha='center')
    plt.text(1, yl, 'Up', ha='center')
    plt.text(5, yl, 'Right', ha='center')
    plt.text(9, yl, 'Down', ha='center')
    plt.text(13, yl, 'Left', ha='center')
    plt.plot(epochs.times, x[:,1,:].T)
    plt.ylim(-yl, yl*1.1)
    plt.savefig(savebase.format('eyemove_eog'), dpi=300, transparent=True)

    ev = epochs.average()
    for m in modalities:
        fig = ev.plot_joint(show=False, picks=m)
        name = 'eyemove_{0}'.format(m)
        fig.savefig(savebase.format(name), dpi=300, transparent=True)
    plt.close('all')

    # Plot blinks
    event_dict = {'blinks': 5}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=2,
                        preload=True)
    x = epochs.get_data()

    plt.figure(figsize=(16,4))
    plt.subplot(121)
    plt.plot(epochs.times, x[:,0,:].T)
    plt.title(raw.info['ch_names'][0])
    plt.subplot(122)
    plt.plot(epochs.times, x[:,1,:].T)
    plt.title(raw.info['ch_names'][1])
    name = 'blink_eog'.format(m)
    plt.savefig(savebase.format(name), dpi=300, transparent=True)

    ev = epochs.average()
    for m in modalities:
        fig = ev.plot_joint(show=False, picks=m)
        name = 'blink_{0}'.format(m)
        fig.savefig(savebase.format(name), dpi=300, transparent=True)
    plt.close('all')

    # Plot swallow
    event_dict = {'swallow': 6}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=2,
                        preload=True)

    ev = epochs.average()
    for m in modalities:
        fig = ev.plot_joint(show=False, picks=m)
        name = 'swallow_{0}'.format(m)
        fig.savefig(savebase.format(name), dpi=300, transparent=True)
    plt.close('all')

    # Plot breathe
    event_dict = {'breath': 7}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.5, tmax=5,
                        preload=True)

    ev = epochs.average()
    for m in modalities:
        fig = ev.plot_joint(show=False, picks=m)
        name = 'breathe_{0}'.format(m)
        fig.savefig(savebase.format(name), dpi=300, transparent=True)
    plt.close('all')

    # Plot shrug
    event_dict = {'shrug': 8}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.5, tmax=3,
                        preload=True)

    ev = epochs.average()
    for m in modalities:
        fig = ev.plot_joint(show=False, picks=m)
        name = 'shrug_{0}'.format(m)
        fig.savefig(savebase.format(name), dpi=300, transparent=True)
    plt.close('all')

    # Plot clench
    if len(np.where(events[:,2]==9)[0]) > 0:
        event_dict = {'clench': 9}
        epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.5, tmax=3,
                            preload=True)

        ev = epochs.average()
        for m in modalities:
            fig = ev.plot_joint(show=False, picks=m)
            name = 'clench_{0}'.format(m)
            fig.savefig(savebase.format(name), dpi=300, transparent=True)
        plt.close('all')

    # Plot button press
    event_dict = {'button_press': 257}
    epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.5, tmax=2,
                        preload=True)

    ev = epochs.average()
    for m in modalities:
        fig = ev.plot_joint(show=False, picks=m)
        name = 'buttonpress_{0}'.format(m)
        fig.savefig(savebase.format(name), dpi=300, transparent=True)
    plt.close('all')

    # Return filenames
    savebase = pathlib.Path(savebase)
    filebase = savebase.parent.name + "/" + savebase.name
    filenames = {
        'plt_eyemove_grad': filebase.format('eyemove_grad'),
        'plt_eyemove_mag': filebase.format('eyemove_mag'),
        'plt_eyemove_eog': filebase.format('eyemove_eog'),
        'plt_blink_grad': filebase.format('blink_grad'),
        'plt_blink_mag': filebase.format('blink_mag'),
        'plt_blink_eog': filebase.format('blink_eog'),
        'plt_swallow_grad': filebase.format('swallow_grad'),
        'plt_swallow_mag': filebase.format('swallow_mag'),
        'plt_breathe_grad': filebase.format('breathe_grad'),
        'plt_breathe_mag': filebase.format('breathe_mag'),
        'plt_shrug_grad': filebase.format('shrug_grad'),
        'plt_shrug_mag': filebase.format('shrug_mag'),
        'plt_clench_grad': filebase.format('clench_grad'),
        'plt_clench_mag': filebase.format('clench_mag'),
        'plt_buttonpress_grad': filebase.format('buttonpress_grad'),
        'plt_buttonpress_mag': filebase.format('buttonpress_mag'),
    }
    return filenames


def print_scan_summary(raw):
    """Print a text summary of an MNE file.
    
    Parameters
    ----------
    raw : :py:class:`mne.io.Raw <mne.io.Raw>`
        MNE Raw object.    
    """
    print('Datafile : {0}'.format(raw.filenames))

    # Project name - set by user during acquisition
    print('Project name: {0}'.format(raw.info['proj_name']))
    print('Experimenter: {0}'.format(raw.info['experimenter']))

    date = raw.info['meas_date']
    print('Data acquired: {0}'.format(date))

    # Duration and sample rate of scan
    print('{0} samples at {1}Hz - {2} seconds'.format(raw.n_times,
                                                      raw.info['sfreq'],
                                                      raw.n_times/raw.info['sfreq']))

    # Number of Head Position Indicator coils
    try:
        nhpi = len(raw.info['hpi_meas'][0]['hpi_coils'])
        print('{0} HPI coils acquired'.format(nhpi))
    except:
        print("No HPI info in fif file")

    # Number of channels acquired
    nchans = raw.info['nchan']
    print('{0} channels acquired'.format(nchans))

    # Breakdown channels into channel types
    chtype = [channel_type(raw.info, c) for c in range(nchans)]
    chs, chcounts = np.unique(chtype, return_counts=True)

    for ii in range(len(chs)):
        print('\t{0}:{1}'.format(chs[ii], chcounts[ii]))

    # Head and coild digitisation points
    dig_codes = ('Cardinal', 'HPI', 'EEG', 'Extra')
    digs = [d['kind'] for d in raw.info['dig']]
    d, dcounts = np.unique(digs, return_counts=True)
    print('Digitisation points')
    for ii in range(len(d)):
        print('\t{0}:{1}'.format(dig_codes[ii], dcounts[ii]))

    # Trigger code events
    ev = mne.find_events(raw, min_duration=5/raw.info['sfreq'], verbose=False)
    ev, evcounts = np.unique(ev[:, 2], return_counts=True)
    print('Events')
    for ii in range(len(ev)):
        print('\t{0}:{1}'.format(ev[ii], evcounts[ii]))

    # Bad segments
    durs = np.array([r['duration'] for r in raw.annotations])
    full_dur = raw.n_times/raw.info['sfreq']
    types = [r['description'] for r in raw.annotations]

    for modality in ['grad', 'mag', 'eeg']:
        inds = [s.find(modality) > 0 for s in types]
        mod_dur = np.sum(durs[inds])
        pc = (mod_dur / full_dur) * 100
        s = 'Modality {0} - {1:02f}/{2} seconds rejected     ({3:02f}%)'
        print(s.format(modality, mod_dur, full_dur, pc))
