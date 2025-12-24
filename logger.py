# ======================== logger.py ========================
"""
Logging functionality
"""
import logging
from datetime import datetime
import os

class AppLogger:
    """Application logger"""
    
    def __init__(self, log_file: str = 'image_processing.log'):
        self.log_file = log_file
        self.logger = self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger configuration"""
        # TODO: Create logger with name 'ImageProcessingApp'
        # TODO: Set logging level to DEBUG
        # TODO: Create file handler for self.log_file
        # TODO: Create console handler
        # TODO: Create formatter with timestamp, name, level, message
        # TODO: Add formatter to handlers
        # TODO: Add handlers to logger
        # TODO: Return configured logger
        pass
    
    def info(self, message: str):
        """Log info message"""
        # TODO: Log message at INFO level
        pass
    
    def error(self, message: str):
        """Log error message"""
        # TODO: Log message at ERROR level
        pass
    
    def warning(self, message: str):
        """Log warning message"""
        # TODO: Log message at WARNING level
        pass
    
    def debug(self, message: str):
        """Log debug message"""
        # TODO: Log message at DEBUG level
        pass
