"""
Logging Utility for Vital Signs Monitoring System
Provides colored console logging and file logging
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

from .config import Config


class LoggerSetup:
    """Central logging configuration"""
    
    _initialized = False
    
    @classmethod
    def setup(
        cls,
        name: str = "vitals",
        level: Optional[str] = None,
        log_file: Optional[str] = None,
        console: bool = True,
        colored: bool = True
    ) -> logging.Logger:
        """
        Set up and return a configured logger.
        
        Args:
            name: Logger name
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            log_file: Optional path to log file
            console: Whether to log to console
            colored: Whether to use colored output (if colorlog installed)
            
        Returns:
            Configured logger instance
        """
        config = Config()
        
        # Get settings from config with defaults
        if level is None:
            level = config.get("logging.level", "INFO")
        if log_file is None:
            log_file = config.get("logging.file")
        if console is None:
            console = config.get("logging.console", True)
        if colored is None:
            colored = config.get("logging.colored", True)
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        # Clear existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
            
            if colored and HAS_COLORLOG:
                # Colored formatter
                formatter = colorlog.ColoredFormatter(
                    "%(log_color)s%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S",
                    log_colors={
                        'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red,bg_white',
                    }
                )
            else:
                # Plain formatter
                formatter = logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                    datefmt="%H:%M:%S"
                )
            
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(config.project_root) / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)  # Log everything to file
            
            file_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        cls._initialized = True
        return logger


def setup_logger(name: str = "vitals") -> logging.Logger:
    """
    Convenience function to get a configured logger.
    
    Args:
        name: Logger name (use module name for hierarchy)
        
    Returns:
        Configured logger
        
    Usage:
        from src.utils import setup_logger
        logger = setup_logger(__name__)
        logger.info("Starting processing...")
    """
    return LoggerSetup.setup(name)


def get_logger(name: str = "vitals") -> logging.Logger:
    """Get an existing logger or create one"""
    return logging.getLogger(name)


# Convenience loggers for different modules
def get_camera_logger() -> logging.Logger:
    return setup_logger("vitals.camera")


def get_model_logger() -> logging.Logger:
    return setup_logger("vitals.model")


def get_api_logger() -> logging.Logger:
    return setup_logger("vitals.api")


if __name__ == "__main__":
    # Test the logger
    logger = setup_logger("test")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    print("\nLogger test complete!")
