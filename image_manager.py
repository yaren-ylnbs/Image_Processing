# ======================== image_manager.py ========================
"""
Image management and processing operations
"""
from PIL import Image
import os

class ImageManager:
    """Manages image loading, processing, and saving"""
    
    def __init__(self):
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        self.image_history = []
        
    def load_image(self, file_path):
        """Load an image from file"""
        try:
            self.image_path = file_path
            self.original_image = Image.open(file_path)
            self.processed_image = self.original_image.copy()
            self.image_history = [self.original_image.copy()]
            return True
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    def apply_processor(self, processor):
        """Apply an image processor"""
        if self.original_image is None:
            raise Exception("Please load an image first!")
        
        try:
            self.processed_image = processor.process(self.original_image)
            self.image_history.append(self.processed_image.copy())
            return True
        except Exception as e:
            raise Exception(f"Error applying operation: {str(e)}")
    
    def reset_image(self):
        """Reset image to original"""
        if self.original_image:
            self.processed_image = self.original_image.copy()
            return True
        return False
    
    def save_image(self, file_path):
        """Save the processed image"""
        if self.processed_image is None:
            raise Exception("No image to save!")
        
        try:
            self.processed_image.save(file_path)
            return True
        except Exception as e:
            raise Exception(f"Error saving image: {str(e)}")
    
    def get_image_info(self):
        """Get image information"""
        if self.original_image:
            return {
                'filename': os.path.basename(self.image_path) if self.image_path else "Unknown",
                'size': self.original_image.size,
                'mode': self.original_image.mode,
                'format': self.original_image.format
            }
        return None
    
    def undo(self):
        """Undo last operation"""
        if len(self.image_history) > 1:
            self.image_history.pop()
            self.processed_image = self.image_history[-1].copy()
            return True
        return False

