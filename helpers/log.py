import logging
from pathlib import Path

LOG_DIR = Path("~/l3_project/.logs").expanduser()
LOG_DIR.mkdir(parents=False, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    # Basic setup mirrors https://docs.python.org/3/howto/logging-cookbook.html#using-logging-in-multiple-modules
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid adding handlers multiple times
        logger.setLevel(logging.DEBUG)

        # create file handler which logs even debug messages
        fh = logging.FileHandler(LOG_DIR / f"{name}.log", mode="a")
        fh.setLevel(logging.DEBUG)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)

        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(pathname)s | %(funcName)s | %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.debug(f"New logger successfully initialised at {fh.baseFilename}")
    else:
        logger.debug(f"Logger successfully returned from {logger.handlers[0].baseFilename}")

    return logger
