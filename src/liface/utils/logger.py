# src/liface/logger_config.py

import logging
import os

LOG_DIR = os.path.join(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ),
    "logs",
)
os.makedirs(LOG_DIR, exist_ok=True)


def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger with a file handler."""
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handler = logging.FileHandler(os.path.join(LOG_DIR, log_file), mode="a")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate logs
    if not logger.handlers:
        logger.addHandler(handler)

    return logger
