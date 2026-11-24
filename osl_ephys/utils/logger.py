"""Logging for osl-ephys: the package logger and a per-session log file."""

import logging
import logging.config
import os
import sys
from contextlib import contextmanager
from datetime import datetime

import yaml

# Initialise logging for this sub-module
osl_logger = logging.getLogger("osl_ephys")
osl_logger.setLevel(logging.WARNING)


default_config = """
version: 1
loggers:
  osl_ephys:
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
    format: '{prefix}%(message)s'
  default:
    format: '[%(asctime)s] {prefix}%(levelname)-8s : %(message)s'
    datefmt: '%H:%M:%S'
  verbose:
    format: '[%(asctime)s] {prefix}%(levelname)s - osl-ephys.%(module)s:%(lineno)s : %(message)s'
    datefmt: '%Y-%m-%d %H:%M:%S'

disable_existing_loggers: true

"""


def set_up(prefix="", log_file=None, level=None, console_format=None, startup=True):
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
    if (len(prefix) > 0) and (console_format != "verbose"):
        prefix = prefix + " : "
    new_config = default_config.format(prefix=prefix, log_file=log_file)
    # Load config to dict
    new_config = yaml.load(new_config, Loader=yaml.FullLoader)

    # Remove log file from dict if not user requested
    if log_file is None:
        new_config["loggers"]["osl_ephys"]["handlers"] = ["console"]
        del new_config["handlers"]["file"]

    # Configure osl_logger with dict
    logging.config.dictConfig(new_config)
    # osl_logger.config.dictConfig(new_config)

    # Customise options
    if level is not None:
        set_level(level)
    if console_format is not None:
        set_format(formatter=console_format, prefix=prefix)

    if startup:
        # Say hello
        osl_logger.info("osl-ephys Logger Started")

    # Print some info
    if log_file is not None:
        osl_logger.info("logging to file: {0}".format(log_file))

    # Attribute to let us know if we have setup the OSL logger
    osl_logger.already_setup = True


def set_format(formatter="brief", prefix="", handler="console"):
    """Change the message format of a handler of the osl-ephys logger.

    Parameters
    ----------
    formatter : {'brief', 'default', 'verbose'}
        Name of a format in the default config.
    prefix : str
        Prefix inserted into the format.
    handler : str
        Name of the handler to change.
    """
    config = yaml.load(
        default_config.format(prefix=prefix, log_file=None), Loader=yaml.FullLoader
    )
    spec = config["formatters"][formatter]
    fmt = logging.Formatter(spec["format"], datefmt=spec.get("datefmt"))
    for h in osl_logger.handlers:
        if h.get_name() == handler:
            h.setFormatter(fmt)


def set_level(level, handler="console"):
    """Set new logging level for osl-ephys module.

    Parameters
    ----------
    level : {'CRITICAL', 'WARNING', 'INFO', 'DEBUG'}
        String indicating new logging level
    handler : str
        The handler to set the level for. Defaults to 'console'.
    """
    osl_logger = logging.getLogger("osl")
    for handler in osl_logger.handlers:
        if handler.get_name() == "console":
            if level in ["INFO", "DEBUG"]:
                osl_logger.info(
                    "osl-ephys osl_logger: handler '{0}' level set to '{1}'".format(
                        handler.get_name(), level
                    )
                )
            handler.setLevel(getattr(logging, level))


def get_level(handler="console"):
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
    osl_logger = logging.getLogger("osl")
    for handler in osl_logger.handlers:
        if handler.get_name() == "console":
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


@contextmanager
def set_logging_level(logger, level):
    """Temporarily change the logging level of a logger.

    Parameters
    ----------
    logger : logging.Logger
        Logger to change.
    level : int
        Logging level to set (e.g. ``logging.WARNING``).
    """
    current_level = logger.getEffectiveLevel()
    try:
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(current_level)


# Loggers whose console handlers are pointed at the session log file while a
# session is being processed
_CAPTURED_LOGGERS = ("", "osl_ephys", "mne")


class SessionLogger:
    """Redirects all stdout/stderr to a per-session log file.

    Progress messages can be printed to screen using the ``log`` method.

    Parameters
    ----------
    session : str
        Session identifier used as a prefix in log messages.
    log_dir : str
        Directory to write log files to.

    Examples
    --------
    >>> with SessionLogger("sub-01_task-rest", log_dir="logs") as logger:
    ...     logger.log("Filtering...")
    ...     raw.resample(250)  # verbose output goes to the log file only
    ...     logger.log("Done.")

    Screen output::

        [09:21:15 sub-01_task-rest] Filtering...
        [09:21:16 sub-01_task-rest] Done.
    """

    def __init__(self, session, log_dir):
        self.session = session
        self.log_dir = log_dir
        self.warnings = []

    def _write_line(self, stream, msg):
        """Write a line to a stream (the screen) and the log file."""
        line = f"[{datetime.now().strftime('%H:%M:%S')} {self.session}] {msg}\n"
        stream.write(line)
        stream.flush()
        self._log_file.write(line)
        self._log_file.flush()

    def log(self, msg):
        """Print a progress message to screen (and log file)."""
        self._write_line(sys.__stdout__, msg)

    def warning(self, msg):
        """Print a warning message to screen (and log file).

        Warnings are kept in ``self.warnings``. They are for a session that
        finishes without everything it was meant to produce.
        """
        self.warnings.append(msg)
        self._write_line(sys.__stderr__, f"WARNING: {msg}")

    def error(self, msg):
        """Print an error message to screen (and log file)."""
        self._write_line(sys.__stderr__, f"ERROR: {msg}")

    def write(self, text):
        if text:
            self._log_file.write(text)
            self._log_file.flush()

    def flush(self):
        self._log_file.flush()

    def __enter__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self._log_file = open(os.path.join(self.log_dir, f"{self.session}.log"), "w")
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

        # Handlers of the logging module hold the stream they were created
        # with, so they are pointed at the log file too. The stream is swapped
        # without flushing the old one, which may be a closed session log
        self._old_streams = []
        for name in _CAPTURED_LOGGERS:
            for handler in logging.getLogger(name).handlers:
                if type(handler) is logging.StreamHandler:
                    self._old_streams.append((handler, handler.stream))
                    handler.stream = self
        return self

    def __exit__(self, *args):
        for handler, stream in self._old_streams:
            handler.stream = stream
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr
        self._log_file.close()
