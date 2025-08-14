import logging
import sys
from logging.handlers import RotatingFileHandler


def setup_logging():
    """Configures logging to file and console."""
    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Ensure logs directory exists
    from pathlib import Path

    Path("logs").mkdir(exist_ok=True)

    # File handler
    log_file = "logs/api.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=1024 * 1024 * 5, backupCount=5
    )  # 5 MB per file
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Add handlers if they don't exist
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
