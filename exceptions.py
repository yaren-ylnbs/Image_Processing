# ======================== exceptions.py ========================
"""
Custom exception classes
"""

class ImageProcessingError(Exception):
    """Base exception for image processing errors"""
    pass

class ImageLoadError(ImageProcessingError):
    """Exception raised when image loading fails"""
    pass

class ImageSaveError(ImageProcessingError):
    """Exception raised when image saving fails"""
    pass

class FilterError(ImageProcessingError):
    """Exception raised when filter application fails"""
    pass

class EnhancementError(ImageProcessingError):
    """Exception raised when enhancement application fails"""
    pass

class InvalidImageError(ImageProcessingError):
    """Exception raised when image is invalid"""
    pass

class UnsupportedFormatError(ImageProcessingError):
    """Exception raised when image format is not supported"""
    pass
