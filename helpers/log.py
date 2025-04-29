import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
import os

# import multiprocess

import helpers.env_var  # must be imported before this file (log) in __init__.py

LOG_DIR = Path(f"{helpers.env_var.get_project_root()}/.logs")
LOG_DIR.mkdir(parents=False, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Returns a logger with the given name that formats logging messages."""

    # Basic setup mirrors https://docs.python.org/3/howto/logging-cookbook.html#using-logging-in-multiple-modules
    active_logger = logging.getLogger(name)

    if not active_logger.handlers:  # avoid adding handlers multiple times
        active_logger.setLevel(logging.DEBUG)

        # create file handler which logs debug messages (and creates a new file for each run of the program)
        log_path = LOG_DIR / f"{name}.log"
        # File opening is deferred until first emit()/logging call by delay=True arg
        fh = RotatingFileHandler(log_path, mode="a", delay=True, backupCount=5)
        # if log_path.is_file():  # check if file already exists (not written to yet by this process since delay=True)
        #     fh.doRollover()  # archive (add .1/.2/etc.) the old log file if it exists

        fh.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s | %(name)-s(id=%(process)d) | %(levelname)7s | "
            "%(filename)s:%(lineno)d | %(funcName)s | %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        active_logger.addHandler(fh)
        active_logger.addHandler(ch)

        active_logger.debug(f"New logger successfully initialised at {fh.baseFilename}")
    else:
        # noinspection PyUnresolvedReferences
        active_logger.debug(f"Logger successfully returned from {active_logger.handlers[0].baseFilename}")

    return active_logger


# exported for modules to import and use
# if multiprocess.parent_process() is None:  # this process is the main process
#     main_logger = get_logger(f"main_{int(time.time())}")
# else:
#     main_logger = get_logger(f"mp_{int(time.time())}")

# Logs are named using the initiation time and process ID.
# This variable can be imported by other modules to share the same logger.
main_logger = get_logger(f"main_{int(time.time())}_{os.getpid()}")
