"""Reporting tool for source reconstruction.

"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
# Mats van Es <mats.vanes@psych.ox.ac.uk>

import os
import os.path as op
from pathlib import Path
from shutil import copy

import pickle
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from tabulate import tabulate
import inspect

from . import preproc_report
from ..source_recon import parcellation, batch
from ..utils.parallel import schedule_or_execute_task

def gen_html_data(config, outdir, subject, reportdir, logger=None, extra_funcs=None, logsdir=None):
    """Generate data for HTML report.

    Parameters
    ----------
    config : dict
        Source reconstruction config.
    outdir : str
        Source reconstruction directory.
    subject : str
        Subject name.
    reportdir : str
        Report directory.
    logger : logging.getLogger
        Logger.
    extra_funcs : list
        List of extra functions to run
    logsdir : str
        Directory the log files were saved into. If None, log files are assumed
        to be in reportdir.replace('report', 'logs')
    """
    outdir = Path(outdir)
    reportdir = Path(reportdir)

    # Make directory for plots contained in the report
    os.makedirs(reportdir, exist_ok=True)
    os.makedirs(reportdir / subject, exist_ok=True)

    # Check if this function has been called before
    if Path(f"{reportdir}/{subject}/data.pkl").exists():
        # Load the data object from last time
        data = pickle.load(open(f"{reportdir}/{subject}/data.pkl", "rb"))

        if "config" in data:
            # Update the config based on this run
            data["config"] = update_config(data["config"], config)
        else:
            data["config"] = config

    # add extra funcs if they exist
    if extra_funcs is not None:
        if 'extra_funcs' not in data.keys():
            data['extra_funcs'] = ""
        for func in extra_funcs:
            data['extra_funcs'] += f"{inspect.getsource(func)}\n\n"

    data["fif_id"] = subject
    data["filename"] = subject

    # What have we done for this subject?
    data["compute_surfaces"] = data.pop("compute_surfaces", False)
    data["coregister"] = data.pop("coregister", False)
    data["beamform"] = data.pop("beamform", False)
    data["minimum_norm"] = data.pop("minimum_norm", False)
    data["parcellate"] = data.pop("parcellate", False)
    data["beamform_and_parcellate"] = data.pop("beamform_and_parcellate", False)
    data["minimum_norm_and_parcellate"] = data.pop("minimum_norm_and_parcellate", False)
    data["fix_sign_ambiguity"] = data.pop("fix_sign_ambiguity", False)

    if data["parcellate"]:
        data["parcellation_filename"] = Path(data["parcellation_file"]).name

    # Copy plots
    if "surface_plots" in data:
        for plot in data["surface_plots"]:
            surface = "surfaces_" + Path(plot).stem
            data[f"plt_{surface}"] = f"{subject}/{surface}.png"
            copy("{}/{}".format(outdir, plot), "{}/{}/{}.png".format(reportdir, subject, surface))

    if "coreg_plot" in data:
        data["plt_coreg"] = f"{subject}/coreg.html" # this might not have been rendered yet (if using dask), so potentially need to copy it over later
        if outdir.__str__() in data['coreg_plot']:
            origfile = data["coreg_plot"]
        else:
            origfile = "{}/{}".format(outdir, data["coreg_plot"])
        schedule_or_execute_task(subject, copy, origfile, "{}/{}/coreg.html".format(reportdir, subject))
        # copy("{}/{}".format(outdir, data["coreg_plot"]), "{}/{}/coreg.html".format(reportdir, subject))

    if "filters_cov_plot" in data:
        data["plt_filters_cov"] = f"{subject}/filters_cov.png"
        copy("{}/{}".format(outdir, data["filters_cov_plot"]), "{}/{}/filters_cov.png".format(reportdir, subject))

    if "filters_svd_plot" in data:
        data["plt_filters_svd"] = f"{subject}/filters_svd.png"
        copy("{}/{}".format(outdir, data["filters_svd_plot"]), "{}/{}/filters_svd.png".format(reportdir, subject))

    if "parc_psd_plot" in data:
        data["plt_parc_psd"] = f"{subject}/parc_psd.png"
        copy("{}/{}".format(outdir, data["parc_psd_plot"]), "{}/{}/parc_psd.png".format(reportdir, subject))

    if "parc_corr_plot" in data:
        data["plt_parc_corr"] = f"{subject}/parc_corr.png"
        copy("{}/{}".format(outdir, data["parc_corr_plot"]), "{}/{}/parc_corr.png".format(reportdir, subject))

    if "parc_freqbands_plot" in data and data["parc_freqbands_plot"] is not None:
        data["plt_parc_freqbands"] = f"{subject}/parc_freqbands.png"
        copy("{}/{}".format(outdir, data["parc_freqbands_plot"]), "{}/{}/parc_freqbands.png".format(reportdir, subject))

    # Logs
    if logsdir is None:
        logsdir = os.path.join(outdir, 'logs')
    
    # guess the log file name
    g = glob(os.path.join(logsdir, f'{subject}*.log'))    
    if len(g)>0:
        for ig in g:
            with open(ig, 'r') as log_file:
                if 'error' in ig:
                    data['errlog'] = log_file.read()
                else:
                    data['log'] = log_file.read()

    # Save data in the report directory
    pickle.dump(data, open(f"{reportdir}/{subject}/data.pkl", "wb"))


def gen_html_page(reportdir):
    """Generate an HTML page from a report directory.

    Parameters
    ----------
    reportdir : str
        Directory to generate HTML report with.
        
    Returns
    -------
    bool
        Whether the report was generated successfully.    
    """
    reportdir = Path(reportdir)

    # Subdirectories which contains plots for each fif file
    subdirs = sorted([d.stem for d in Path(reportdir).iterdir() if d.is_dir()])

    # Load HTML data
    data = []
    for subdir in subdirs:
        subdir = Path(subdir)
        # Just generate the html page with the successful runs
        try:
            data.append(pickle.load(open(reportdir / subdir / "data.pkl", "rb")))
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
    panel_template = preproc_report.load_template('src_subject_panel')

    for i in range(total):
        panels.append(panel_template.render(data=data[i]))

    # Hyperlink to each panel on the page
    filenames = ""
    for i in range(total):
        filename = Path(data[i]["filename"]).name
        filenames += "{0}. {1}<br>".format(i + 1, filename)

    # Render the full page
    page_template = preproc_report.load_template('subject_report')
    page = page_template.render(panels=panels, filenames=filenames)

    # Write the output file
    outpath = Path(reportdir) / 'subject_report.html'
    with open(outpath, 'w') as f:
        f.write(page)

    return True


def gen_html_summary(reportdir, logsdir=None):
    """Generate an HTML summary from a report directory.

    Parameters
    ----------
    reportdir : str
        Directory to generate HTML summary report with.
        logsdir: str
    Directory the log files were saved into. If None, log files are assumed
        to be in reportdir.replace('report', 'logs')
        
    Returns
    -------
    bool
        Whether the report was generated successfully.
    """
    reportdir = Path(reportdir)

    # Subdirectories which contains plots for each fif file
    subdirs = sorted([d.stem for d in Path(reportdir).iterdir() if d.is_dir()])

    # Load HTML data
    subject_data = []
    for subdir in subdirs:
        subdir = Path(subdir)
        # Just generate the html page with the successful runs
        try:
            subject_data.append(pickle.load(open(reportdir / subdir / "data.pkl", "rb")))
        except:
            pass

    total = len(subject_data)
    if total == 0:
        return False

    # Data used in the summary report
    data = {}
    data["total"] = total
    data["config"] = subject_data[0]["config"]
    if "extra_funcs" in subject_data[0]:
        data["extra_funcs"] = subject_data[0]["extra_funcs"] 
    data["coregister"] = subject_data[0]["coregister"]
    data["beamform"] = subject_data[0]["beamform"]
    data["beamform_and_parcellate"] = subject_data[0]["beamform_and_parcellate"]
    data["minimum_norm"] = subject_data[0]["minimum_norm"]
    data["minimum_norm_and_parcellate"] = subject_data[0]["minimum_norm_and_parcellate"]
    data["fix_sign_ambiguity"] = subject_data[0]["fix_sign_ambiguity"]

    if data["coregister"]:
        subjects = np.array([d["filename"] for d in subject_data])

        fid_err_table = pd.DataFrame()
        fid_err_table["Session ID"] = [subject_data[i]["fif_id"] for i in range(len(subject_data))]
        if len(subject_data[0]['fid_err'])==4:
            for i_err, hdr in enumerate(["Nasion", "LPA", "RPA", "RMS(Headshape-MRI)"]):
                fid_err_table[hdr] = [np.round(subject_data[i]['fid_err'][i_err], decimals=2) if 'fid_err' in subject_data[i].keys() else None for i in range(len(subject_data))]   
        else:
            for i_err, hdr in enumerate(["Nasion", "LPA", "RPA", "RMS(Headshape-MRI)"]):
                fid_err_table[hdr] = [np.round(subject_data[i]['fid_err'][i_err], decimals=2) if 'fid_err' in subject_data[i].keys() else None for i in range(len(subject_data))]
        fid_err_table.index += 1 # Start indexing from 1
        data['coreg_table'] = fid_err_table.to_html(classes="display", table_id="coreg_tbl")

    # Create plots
    os.makedirs(f"{reportdir}/summary", exist_ok=True)

    data["plt_config"] = plot_config(data["config"], reportdir)

    if "parcellation_file" in subject_data[0]:
        data["parcellation_filename"] = subject_data[0]["parcellation_filename"]
        data["plt_parc"] = plot_parcellation(
            subject_data[0]["parcellation_file"],
            reportdir,
        )

    if data["fix_sign_ambiguity"]:
        data["template"] = subject_data[0]["template"]
        metrics = np.array([d["metrics"] for d in subject_data if "metrics" in d])
        data["plt_sflip"] = plot_sign_flipping_results(metrics, reportdir)

        flip_table = pd.DataFrame()
        flip_table["Session ID"] = [subject_data[i]["fif_id"] for i in range(len(subject_data))]
        flip_table["Correlation before flipping "] = np.round(metrics[:,0], decimals=3)
        flip_table["Correlation after flipping "] = np.round(metrics[:,1:].max(axis=1), decimals=3)
        flip_table["Correlation change (after-before) "] = np.round(flip_table["Correlation after flipping "]-flip_table["Correlation before flipping "], decimals=3)
        flip_table.index += 1 # Start indexing from 1
        data['signflip_table'] = flip_table.to_html(classes="display", table_id="signflip_tbl")

    # log files
    if logsdir is None:
        logsdir = reportdir._str.replace('src_report', 'logs')
        
    if os.path.exists(os.path.join(logsdir, 'batch_src.log')):
        with open(os.path.join(logsdir, 'batch_src.log'), 'r') as log_file:
            data['batchlog'] = log_file.read()
    
    g = glob(os.path.join(logsdir, '*.error.log'))    
    if len(g)>0:
        data['errlog'] = {}
        for ig in sorted(g):
            with open(ig, 'r') as log_file:
                data['errlog'][ig.split('/')[-1].split('.error.log')[0]] = log_file.read()
            
    # Create panel
    panel_template = preproc_report.load_template('src_summary_panel')
    panel = panel_template.render(data=data)

    # List of filenames
    filenames = ""
    for i in range(total):
        filename = Path(subject_data[i]["filename"]).name
        filenames += "{0}. {1}<br>".format(i + 1, filename)

    # Render the full page
    page_template = preproc_report.load_template('summary_report')
    page = page_template.render(panel=panel, filenames=filenames)

    # Write the output file
    outpath = Path(reportdir) / 'summary_report.html'
    with open(outpath, 'w') as f:
        f.write(page)

    return True


def plot_config(config, reportdir):
    """Plots a config flowchart.

    Parameters
    ----------
    config : dict
        Config to plot.
    reportdir : str
        Path to report directory. We will save the plot in this directory.

    Returns
    -------
    path : str
        Path to plot.
    """

    fig = plt.figure(figsize=(8, 6))
    plt.subplots_adjust(top=0.95, bottom=0.05)
    ax = plt.subplot(111, frame_on=False)
    ax.set_xticks([])
    ax.set_yticks([])

    stage_height = 1 / (1 + len(config["source_recon"]))
    stagecol = "wheat"
    startcol = "red"

    box = dict(boxstyle="round", facecolor=stagecol, alpha=1, pad=0.3)
    startbox = dict(boxstyle="round", facecolor=startcol, alpha=1)
    font = {
        "family": "serif",
        "color": "k",
        "weight": "normal",
        "size": 14,
    }

    stages = [{"input": ""}, *config["source_recon"], {"output": ""}]
    stage_str = "$\\bf{{{0}}}$ {1}"

    ax.arrow(
        0.5, 1, 0.0, -1, fc="k", ec="k", head_width=0.045,
        head_length=0.035, length_includes_head=True,
    )

    for idx, stage in enumerate(stages):
        method, userargs = next(iter(stage.items()))

        method = method.replace("_", "\_")
        if method in ["input", "output"]:
            b = startbox
        else:
            b = box
            method = method + ":"

        ax.text(
            0.5,
            1 - stage_height * idx,
            stage_str.format(method, str(userargs)[1:-1]),
            ha="center",
            va="center",
            bbox=b,
            fontdict=font,
            wrap=True,
        )

    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.25, 0.75)

    fig.savefig(f"{reportdir}/summary/config.png", dpi=300, transparent=True)
    plt.close(fig)

    return f"summary/config.png"


def plot_parcellation(parcellation_file, reportdir):
    """Plot parcellation.

    Parameters
    ----------
    parcellation_file : str
        Path to parcellation file.
    reportdir : str
        Path to report directory. We will save the plot in this directory.

    Returns
    -------
    path : str
        Path to plot.
    """
    output_file = reportdir / "summary/parc.png"
    parcellation.plot_parcellation(parcellation_file, output_file=output_file)
    return f"summary/parc.png"


def plot_sign_flipping_results(metrics, reportdir):
    """Plot sign flipping results.

    Parameters
    ----------
    metrics : np.ndarray
        Sign flipping metrics. Shape is (n_subjects, n_iter + 1).
    reportdir : str
        Path to report directory. We will save the plot in this directory.

    Returns
    -------
    path : str
        Path to plot.
    """
    output_file = reportdir / "summary/sflip.png"
    fig, ax = plt.subplots()
    for i in range(metrics.shape[-1]):
        if i==0:
            label = "Before flipping"
        else:
            label = f"Init {i}"
        ax.plot(
            range(1, metrics.shape[0] + 1),
            metrics[:, i],
            label=label,
        )
    ax.legend()
    ax.set_xlabel("Subject")
    ax.set_ylabel("Correlation w.r.t. template subject covariance")
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    return f"summary/sflip.png"


def add_to_data(data_file, info):
    """Adds info to a dictionary containing info for the source recon report.

    Parameters
    ----------
    data_file : str
        Path to pickle file containing the data dictionary.
    info : dict
        Info to add.
    """
    data_file = Path(data_file)
    if data_file.exists():
        data = pickle.load(open(data_file, "rb"))
    else:
        data_file.parent.mkdir(parents=True)
        data = {}
    data.update(info)
    pickle.dump(data, open(data_file, "wb"))


def update_config(old_config, new_config):
    """Merge/update a config.

    Parameters
    ----------
    old_config : dict
        Old config.
    new_config : dict
        New config.

    Returns
    -------
    config : dict
        Merge/updated config.
    """
    if not isinstance(old_config, dict):
        old_config = batch.load_config(old_config)
    old_stages = []
    for stage in old_config["source_recon"]:
        for k, v in stage.items():
            old_stages.append(k)
    new_stages = []
    for stage in new_config["source_recon"]:
        for k, v in stage.items():
            new_stages.append(k)
    for i, new_stage in enumerate(new_stages):
        if new_stage not in old_stages:
            old_config["source_recon"].append(new_config["source_recon"][i])
        else:
            for j in range(len(old_config["source_recon"])):
                if new_stage in old_config["source_recon"][j]:
                    old_config["source_recon"][j][new_stage] = (
                        new_config["source_recon"][i][new_stage]
                    )
    return old_config
