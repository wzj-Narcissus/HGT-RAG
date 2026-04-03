"""Centralised logging configuration for the HGT pipeline.

All modules obtain a logger via ``get_logger(__name__)`` so that log output
is uniformly formatted and can be controlled with a single call to
``logging.basicConfig`` or by setting the log level on the root logger.
"""
import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a consistent timestamp + level + name format.

    Handlers are only attached once, so calling ``get_logger`` multiple times
    for the same ``name`` is safe.

    Args:
        name:  Logger name (typically ``__name__`` of the calling module).
        level: Logging level (default ``logging.INFO``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

