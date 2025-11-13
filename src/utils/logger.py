"""
Logging Configuration
=====================

Centralized logging setup for the LSTM Frequency Extraction System.
Provides consistent logging across all modules with configurable levels and formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console output for better readability.
    """

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }

    def format(self, record):
        """Add color to log level name."""
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname:8s}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Configure and return a logger with both console and file handlers.

    Args:
        name: Logger name (usually __name__ of the module)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional path to log file. If None, uses default 'logs/{name}.log'
        console_output: Whether to output to console (default: True)

    Returns:
        logging.Logger: Configured logger instance

    Example:
        >>> from src.utils.logger import setup_logger
        >>> logger = setup_logger(__name__, level="DEBUG")
        >>> logger.info("Starting data generation")
        >>> logger.warning("Low memory available")
        >>> logger.error("Failed to load model", exc_info=True)

    Raises:
        ValueError: If invalid log level provided
        PermissionError: If cannot create log directory
    """
    # Validate log level
    valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    level = level.upper()
    if level not in valid_levels:
        raise ValueError(
            f"Invalid log level '{level}'. Must be one of {valid_levels}"
        )

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    logger.propagate = False  # Prevent duplicate logs

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level))
        console_format = ColoredFormatter(
            fmt='%(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    # File handler without colors
    if log_file or level == 'DEBUG':  # Always log DEBUG to file
        if log_file is None:
            # Default log directory
            log_dir = Path('outputs/logs')
            try:
                log_dir.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                raise PermissionError(
                    f"Cannot create log directory '{log_dir}': {e}"
                ) from e

            # Generate timestamped log file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = log_dir / f"{name.replace('.', '_')}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # Always DEBUG for files
        file_format = logging.Formatter(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one with default settings.

    Args:
        name: Logger name (usually __name__)

    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Logger doesn't exist, create with defaults
        logger = setup_logger(name)
    return logger


# Module-level logger for this file
logger = setup_logger(__name__)


if __name__ == '__main__':
    # Demonstration
    print("=== Logger Demonstration ===\n")

    # Create test logger
    test_logger = setup_logger('test_module', level='DEBUG', log_file='outputs/logs/test.log')

    # Test different log levels
    test_logger.debug("This is a DEBUG message - detailed information")
    test_logger.info("This is an INFO message - general information")
    test_logger.warning("This is a WARNING message - something unexpected")
    test_logger.error("This is an ERROR message - serious problem")
    test_logger.critical("This is a CRITICAL message - program may crash")

    print("\n✅ Logs written to outputs/logs/test.log")
    print("✅ Console output shows colored levels")
