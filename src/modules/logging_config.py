# src/logging_config.py

import logging
import os
from logging.handlers import RotatingFileHandler

# Placeholder for the log directory path
LOG_DIR = "logs"  # <-- Replace this with your desired log folder path

# Ensure the log directory exists
os.makedirs(LOG_DIR, exist_ok=True)

def get_logger(module_name: str) -> logging.Logger:
    """
    Creates and returns a logger for the specified module.
    Each logger writes to a module-specific log file and a common log file.
    
    Args:
        module_name (str): The name of the module (typically __name__).
    
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)  # Set the desired logging level

    if not logger.handlers:
        # Formatter for log messages
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Module-specific log file handler
        module_log_path = os.path.join(LOG_DIR, f"{module_name}.log")
        module_handler = RotatingFileHandler(
            module_log_path, maxBytes=5*1024*1024, backupCount=5
        )
        module_handler.setLevel(logging.DEBUG)
        module_handler.setFormatter(formatter)
        logger.addHandler(module_handler)

        # Common log file handler
        common_log_path = os.path.join(LOG_DIR, "common.log")
        common_handler = RotatingFileHandler(
            common_log_path, maxBytes=10*1024*1024, backupCount=5
        )
        common_handler.setLevel(logging.DEBUG)
        common_handler.setFormatter(formatter)
        logger.addHandler(common_handler)

        # Console handler (optional)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
