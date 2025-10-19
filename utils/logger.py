"""
Logging configuration for Missing Person AI system.

This module provides structured logging setup using loguru with
console and file output, rotation, and different log levels.
"""

import sys
import os
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_level: str = "INFO",
    log_file: str = "logs/app.log",
    rotation: str = "1 day",
    retention: str = "30 days",
    compression: str = "zip"
) -> None:
    """
    Setup logger with console and file output.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        rotation: Log rotation interval
        retention: Log retention period
        compression: Compression format for rotated logs
    """
    # Remove default logger
    logger.remove()
    
    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Console logging with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # File logging without colors
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
               "{name}:{function}:{line} - {message}",
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        enqueue=True  # Thread-safe logging
    )
    
    logger.info(f"Logger initialized with level {log_level}")
    logger.info(f"Log file: {log_file}")


def get_logger(name: Optional[str] = None):
    """
    Get logger instance.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    if name:
        return logger.bind(name=name)
    return logger


# Initialize logger on import
setup_logger()
