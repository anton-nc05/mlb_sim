# File: mlb_sim/utils/logging.py
"""
Logging configuration.
"""

import logging

def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger.

    Args:
        name: Name of the logger.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
