# ======================== base_classes.py ========================
"""
Base classes for image processing operations
"""
from abc import ABC, abstractmethod

class ImageProcessor(ABC):
    """Base class for image processing operations"""
    
    @abstractmethod
    def process(self, image):
        """Image processing function - must be implemented in subclasses"""
        pass

class Filter(ImageProcessor):
    """Base class for filter operations"""
    
    def __init__(self, name):
        #Initialize filter with name
        self.name = name
    
    def __str__(self):
        #Return string representation of filter
        return f"{self.name} Filter"

class ImageEnhancement(ImageProcessor):
    """Base class for image enhancement operations"""
    
    def __init__(self, name, factor):
        #Initialize enhancement with name and factor
        self.name = name
        self.factor = factor
    
    def __str__(self):
        #Return string representation of enhancement
        return f"{self.name} Enhancement (Factor: {self.factor})"