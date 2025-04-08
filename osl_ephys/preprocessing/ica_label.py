"""
OSL ICA LABEL
Tool for interactive ICA component labeling and rejection. 
Works with command line arguments or as a function call.

"""

# Authors: Mats van Es <mats.vanes@psych.ox.ac.uk>

import sys
import os
import mne
import numpy as np
import pickle
import logging
import traceback
from glob import glob
from copy import deepcopy
from time import localtime, strftime
from matplotlib import pyplot as plt
from osl_ephys.preprocessing.plot_ica import plot_ica
from osl_ephys.report import plot_bad_ica
from osl_ephys.report.preproc_report import gen_html_page, gen_html_summary
from osl_ephys.utils import logger as osl_logger

logger = logging.getLogger(__name__)


def ica_label(data_dir, subject, reject=None, interactive=True):
    """Data bookkeeping and wrapping plot_ica.
    
    Parameters
    ----------
    data_dir : str
        Path to processed (M/EEG) data.
    subject : str
        Subject/session specific data directory name
    reject : bool or str
        If 'all', reject all components (previously labeled and newly
        labeled). If 'manual', reject only manually labeled components. If 
        None (default), only save the ICA data; don't reject any 
        components from the M/EEG data.
    """
    if isinstance(subject, list):
        for sub in subject:
            ica_label(data_dir, sub, reject=reject, interactive=interactive)
        return
    
    plt.ion()
    
    # define data paths based on osl-ephys data structure
    preproc_file = os.path.join(data_dir, subject, subject + '_preproc-raw.fif')
    ica_file = os.path.join(data_dir, subject, subject + '_ica.fif')
    report_dir_base =  os.path.join(data_dir, 'preproc_report')
    report_dir = os.path.join(report_dir_base, subject)
    logs_dir = os.path.join(data_dir, 'logs')
    logfile = os.path.join(logs_dir, subject + '_preproc.log')
    
    # setup loggers
    mne.utils._logging.set_log_file(logfile, overwrite=False)
    osl_logger.set_up(prefix=subject, log_file=logfile, level="INFO")
    mne.set_log_level("INFO")
    logger = logging.getLogger(__name__)
    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    logger.info("{0} : Starting osl-ephys Processing".format(now))
    try:                               
        logger.info('Importing {0}'.format(preproc_file))
        raw = mne.io.read_raw(preproc_file, preload=True)
        logger.info('Importing {0}'.format(ica_file))
        ica = mne.preprocessing.read_ica(ica_file)
        
        # keep these for later
        if reject=='manual':
            exclude_old = deepcopy(ica.exclude)

        # interactive components plot
        if interactive:
            logger.info('INTERACTIVE ICA LABELING')
            plot_ica(ica, raw, block=True, stop=30)
            plt.pause(0.1)

        if reject == 'all' or reject == 'manual':
            logger.info("Removing {0} labelled components from the data".format(reject))
            if reject == 'all' or interactive is False:
                ica.apply(raw)
            elif reject == 'manual':
                # we need to make sure we don't remove components that 
                # were already removed before
                new_ica = deepcopy(ica)
                new_ica.exclude = np.setdiff1d(ica.exclude, exclude_old)
                new_ica.apply(raw)
                
            logger.info("Saving preprocessed data")
            raw.save(preproc_file, overwrite=True)
        else:
            logger.info("Not removing any components from the data")
        
        logger.info("Saving ICA data")

        # make sure the format is correct, otherwise errors will occur
        for key in ica.labels_.keys():
            ica.labels_[key] = list(ica.labels_[key])

        ica.save(ica_file, overwrite=True)

        # if reject is not None:
        logger.info("Attempting to update report")
        
        savebase = os.path.join(report_dir, "{0}.png")
        logger.info("Assuming report directory: {0}".format(report_dir))
        if os.path.exists(os.path.join(report_dir, "ica.png")) or os.path.exists(os.path.join(report_dir, "data.pkl")):
            logger.info("Generating ICA plot")
            _ = plot_bad_ica(raw, deepcopy(ica), savebase)

            # try updating the report data
            logger.info("Updating data.pkl")
            data = pickle.load(open(os.path.join(report_dir, "data.pkl"), 'rb'))
            if 'plt_ica' not in data.keys():
                data['plt_ica'] = os.path.join(report_dir.split("/")[-1], "ica.png")

            # update number of bad components
            data['ica_ncomps_rej'] = len(ica.exclude)

            data['ica_ncomps_rej_ecg'] = [len(ica.labels_['ecg']) if 'ecg' in ica.labels_.keys() else 'N/A'][0]
            data['ica_ncomps_rej_eog'] = [len(ica.labels_['eog']) if 'eog' in ica.labels_.keys() else 'N/A'][0]

            # save data
            pickle.dump(data, open(os.path.join(report_dir, "data.pkl"), 'wb'))

            # gen html pages
            logger.info("Generating subject_report.html")
            gen_html_page(report_dir_base)
            logger.info("Generating summary_report.html")
            gen_html_summary(report_dir_base)
            logger.info("Successfully updated report")
                
    except Exception as e:
        logger.critical("**********************")
        logger.critical("* PROCESSING FAILED! *")
        logger.critical("**********************")
        ex_type, ex_value, ex_traceback = sys.exc_info()
        logger.error("osl_ica_label")
        logger.error(ex_type)
        logger.error(ex_value)
        logger.error(traceback.print_tb(ex_traceback))
        with open(logfile.replace(".log", ".error.log"), "w") as f:
            f.write("OSL-EPHYS PREPROCESSING CHAIN failed at: {0}".format(now))
            f.write("\n")
            f.write('Processing filed during stage : "{0}"'.format('osl_ica_label'))
            f.write(str(ex_type))
            f.write("\n")
            f.write(str(ex_value))
            f.write("\n")
            traceback.print_tb(ex_traceback, file=f)
                
    now = strftime("%Y-%m-%d %H:%M:%S", localtime())
    logger.info("{0} : Processing Complete".format(now))


def main(argv=None):
    """
    Command-line interface for ica_label.
    
    Parameters
    ----------
    argv : list
        List of strings to be parsed as command-line arguments. If None, 
        sys.argv will be used.
        
    Example
    -------
    From the command line (in the osl-ephys environment), use as follows:

    osl_ica_label reject_argument /path/to/processed_data subject_name

    The `reject_argument` specifies whether to reject 'all' selected components from the data, only
    the 'manual' rejected, or None (and only save the ICA object, without rejecting components). 
    The `subject_name` should be the name of the subject directory in the processed data directory. 
    The /path/to/processed_data can be omitted when the command is run from the processed data directory.
    If both the subject_name and directory are omitted, the script will attempt to process all subjects in the 
    processed data directory.
    For example:
    
    osl_ica_label manual /path/to/proc_dir sub-001_run01
    
    or:
    
    osl_ica_label all sub-001_run01
    
    Then use the GUI to label components (click on the time course to mark, use 
    number keys to label marked components as specific artefacts, and use
    the arrow keys to navigate. Close the plot.
    all/manual/None components will be removed from the M/EEG data and saved. The 
    ICA data will be saved with the new labels. If the report directory is specified
    or in the assumed osl-ephys directory structure, the subject report and log file is updated.

    """

    if argv is None:
        argv = sys.argv[1:]
    
    reject = argv[0]
    if reject == 'None':
        reject = None
    
    if len(argv)<3:
        data_dir = os.getcwd()
        if len(argv)==2:
            subject = argv[1]
        else:
            g = sorted(glob(os.path.join(f"{data_dir}", '*', '*_ica.fif')))
            subject = [f.split('/')[-2] for f in g]
            # batch log
            logs_dir = os.path.join(data_dir, 'logs')
            logfile = os.path.join(logs_dir, 'osl_batch.log')
            osl_logger.set_up(log_file=logfile, level="INFO", startup=False)
            logger.info('Starting osl-ephys ICA Batch Processing')
            logger.info('Running osl_ica_label on {0} subjects with reject={1}'.format(len(subject), str(reject)))
    else:
        data_dir = argv[1]
        subject = argv[2]
    
    ica_label(data_dir=data_dir, subject=subject, reject=reject)


def apply(argv=None):
    """
    Command-line function for removing all labeled components from the data.
    
    Parameters
    ----------
    argv : list
        List of strings to be parsed as command-line arguments. If None, 
        sys.argv will be used.
        
    Example
    -------
    From the command line (in the osl-ephys environment), use as follows:

    osl_ica_apply /path/to/processed_data subject_name

    The `subject_name` should be the name of the subject directory in the processed data directory. If omitted, 
    the script will attempt to process all subjects in the processed data directory. The /path/to/processed_data
    can also be omitted when the command is run from the processed data directory (only when processing all subjects).

    For example:
    
    osl_ica_apply /path/to/proc_dir sub-001_run01
    
    or:
    
    osl_ica_apply
    
    Then use the GUI to label components (click on the time course to mark, use 
    number keys to label marked components as specific artefacts, and use
    the arrow keys to navigate. Close the plot.
    all/manual/None components will be removed from the M/EEG data and saved. The 
    ICA data will be saved with the new labels. If the report/logs directories are 
    in the assumed osl-ephys directory structure, the subject report and log file are updated.

    """

    if argv is None and len(sys.argv)>1:
        argv = sys.argv[1:]
        
    subject = None   
    if argv is None:
        data_dir = os.getcwd()    
    else:
        data_dir = argv[0]
        if len(argv)==2:
            subject = argv[1]

    if subject is None:
        g = sorted(glob(os.path.join(f"{data_dir}", '*', '*_ica.fif')))
        subject = [f.split('/')[-2] for f in g]
        
        # batch log
        logs_dir = os.path.join(data_dir, 'logs')
        logfile = os.path.join(logs_dir, 'osl_batch.log')
        osl_logger.set_up(log_file=logfile, level="INFO", startup=False)
        logger.info('Starting osl-ephys ICA Batch Processing')
        logger.info('Running osl_ica_apply on {0} subjects'.format(len(subject)))
    
    ica_label(data_dir=data_dir, subject=subject, reject='all', interactive=False)



if __name__ == '__main__':
    main()
