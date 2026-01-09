"""Logging configuration for {{ cookiecutter.package_name }} using loguru."""

from loguru import logger
import sys


def setup_logging(log_level: str = "INFO"):
    """Setup logging with the specified log level using loguru."""
    # Remove default logger to customize it
    logger.remove()
    
    # Add a handler with the specified log level
    logger.add(
        sys.stdout, 
        format="{time} | {level} | {name}:{line} - {message}", 
        level=log_level.upper()
    )
    
    # Return the logger instance for use in other modules
    return logger


# Global logger instance
default_logger = setup_logging()


def get_logger():
    """Get the configured logger instance."""
    return default_logger