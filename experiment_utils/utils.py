"""
Collection of helper methods. These should be fully generic and make no
assumptions about the format of input data.
"""

import logging


def turn_off_package_logger(package: str) -> None:
    """ "
    Turn off logging for a specific package.

    :param package: The name of the package to turn off logging for.
    """
    logger = logging.getLogger(package)
    logger.setLevel(logging.ERROR)
    logger.handlers = [logging.NullHandler()]


def get_logger(name: str) -> logging.Logger:
    """ "
    Get a logger with the specified name.

    :param name: The name of the logger.
    :return: The logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[::-1]:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt="%(asctime)s %(message)s", datefmt="%d/%m/%Y %I:%M:%S %p")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


def log_and_raise_error(logger: logging.Logger, message: str, exception_type: type[Exception] = ValueError) -> None:
    """ "
    Logs an error message and raises an exception of the specified type.

    :param message: The error message to log and raise.
    :param exception_type: The type of exception to raise (default is ValueError).
    """

    logger.error(message)
    raise exception_type(message)
