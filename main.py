import logging
from pathlib import Path

import datasets.dataset_core

# Basic setup mirrors https://docs.python.org/3/howto/logging-cookbook.html#using-logging-in-multiple-modules
logger = logging.getLogger("projectLog")
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(Path("./.logs/core.log"))
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

logger.debug("Logger successfully initialised.")

if __name__ == "__main__":
    print(datasets.dataset_core.get_dataset_root())
