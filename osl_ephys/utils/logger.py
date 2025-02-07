"""Logging module for OSL

Heavily inspired by logging in OSL.
"""

# Authors: Andrew Quinn <a.quinn@bham.ac.uk>

import yaml
import logging

# Initialise logging for this sub-module
osl_logger = logging.getLogger("osl")
osl_logger.setLevel(logging.WARNING)

#%% ------------------------------------------------------------


default_config = """
version: 1
loggers:
  osl:
    level: DEBUG
    handlers: [console, file]
    propagate: false

handlers:
  console:
    class : logging.StreamHandler
    formatter: brief
    level   : DEBUG
    stream  : ext://sys.stdout
  file:
    class : logging.handlers.RotatingFileHandler
    formatter: verbose
    filename: {log_file}
    backupCount: 3
    maxBytes: 102400

formatters:
  brief:
    format: '{prefix} %(message)s'
  default:
    format: '[%(asctime)s] {prefix} %(levelname)-8s : %(message)s'
    datefmt: '%H:%M:%S'
  verbose:
    format: '[%(asctime)s] {prefix} - %(levelname)s - osl-ephys.%(module)s:%(lineno)s : %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'

disable_existing_loggers: true

"""


def set_up(prefix='', log_file=None, level=None, console_format=None, startup=True):
    """Initialise the osl-ephys module osl_logger.

    Parameters
    ----------
    prefix : str
        Optional prefix to attach to osl_logger output
    log_file : str
        Optional path to a log file to record osl_logger output
    level : {'CRITICAL', 'WARNING', 'INFO', 'DEBUG'}
        String indicating initial logging level
    console_format : str
        Formatting string for console logging.

    """
    # Format config with user options
    if (len(prefix) > 0) and (console_format != 'verbose'):
        prefix = prefix + ' :'
    new_config = default_config.format(prefix=prefix, log_file=log_file)
    # Load config to dict
    new_config = yaml.load(new_config, Loader=yaml.FullLoader)

    # Remove log file from dict if not user requested
    if log_file is None:
        new_config['loggers']['osl']['handlers'] = ['console']
        del new_config['handlers']['file']

    # Configure osl_logger with dict
    logging.config.dictConfig(new_config)
    #osl_logger.config.dictConfig(new_config)

    # Customise options
    if level is not None:
        set_level(level)
    if console_format is not None:
        set_format(formatter=console_format, prefix=prefix)

    if startup:
        # Say hello
        osl_logger.info('osl-ephys Logger Started')

    # Print some info
    if log_file is not None:
        osl_logger.info('logging to file: {0}'.format(log_file))

    # Attribute to let us know if we have setup the OSL logger
    osl_logger.already_setup = True


def set_level(level, handler='console'):
    """Set new logging level for osl-ephys module.
    
    Parameters
    ----------
    level : {'CRITICAL', 'WARNING', 'INFO', 'DEBUG'}
        String indicating new logging level
    handler : str
        The handler to set the level for. Defaults to 'console'.
    """
    osl_logger = logging.getLogger('osl')
    for handler in osl_logger.handlers:
        if handler.get_name() == 'console':
            if level in ['INFO', 'DEBUG']:
                osl_logger.info("osl-ephys osl_logger: handler '{0}' level set to '{1}'".format(handler.get_name(), level))
            handler.setLevel(getattr(logging, level))


def get_level(handler='console'):
    """Return current logging level for osl-ephys module.
    
    Parameters
    ----------
    handler : str
        The handler to get the level for. Defaults to 'console'.
        
    Returns
    -------
    level : {'CRITICAL', 'WARNING', 'INFO', 'DEBUG'}
        String indicating current logging level
    
    """
    osl_logger = logging.getLogger('osl')
    for handler in osl_logger.handlers:
        if handler.get_name() == 'console':
            return handler.level


def log_or_print(msg, warning=False):
    """Execute logger.info if an OSL logger has been setup, otherwise print.

    Parameters
    ----------
    msg : str
        Message to log/print.
    warning : bool
        Is the msg a warning? Defaults to False, which will print info.
    """
    if warning:
        msg = f"WARNING: {msg}"
    if hasattr(osl_logger, "already_setup"):
        osl_logger.info(msg)
    else:
        print(msg)
