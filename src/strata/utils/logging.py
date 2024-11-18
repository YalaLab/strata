import logging
import sys


def set_loglevel(debug):
    loglevel = logging.INFO if debug else logging.WARN
    logger.setLevel(loglevel)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(loglevel)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False


# Set it to debug by default. Will override in main.
logger = logging.getLogger("main")
set_loglevel(debug=True)
