import os
import logging
import sys
from datetime import datetime
from typing import Optional

def setup_logging(
    log_dir: str = 'logs',
    log_level: int = logging.INFO,
    log_format: Optional[str] = None,
    console_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration with both file and console handlers.

    Args:
        log_dir: Directory to store log files
        log_level: Logging level (default: INFO)
        log_format: Format for file logs (default: detailed format)
        console_format: Format for console logs (default: brief format)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('quantum_mri')
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Set default formats if not provided
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if console_format is None:
        console_format = '%(levelname)s: %(message)s'

    # Create formatters
    file_formatter = logging.Formatter(log_format)
    console_formatter = logging.Formatter(console_format)

    # File handler (detailed logging)
    log_file = os.path.join(
        log_dir,
        f'quantum_mri_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Keep DEBUG for file
    file_handler.setFormatter(file_formatter)

    # Console handler (brief logging)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initial message
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name.

    Args:
        name: Optional name for the logger (default: 'quantum_mri')

    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name or 'quantum_mri')
