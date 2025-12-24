# ======================== image_manager.py ========================
"""
Image management and processing operations
"""
from PIL import Image
import os

class ImageManager:
    """Manages image loading, processing, and saving"""
    
    def __init__(self):
        # TODO: Initialize instance variables:
        # - original_image (None)
        # - processed_image (None)
        # - image_path (None)
        # - image_history (empty list)
        pass
        
    def load_image(self, file_path):
        """Load an image from file"""
        # TODO: Try to load image from file_path using Image.open
        # TODO: Store file_path in self.image_path
        # TODO: Store loaded image in self.original_image
        # TODO: Create a copy for self.processed_image
        # TODO: Initialize self.image_history with a copy of original_image
        # TODO: Return True on success
        # TODO: Handle exceptions and raise with descriptive message
        pass
    
    def apply_processor(self, processor):
        """Apply an image processor"""
        # TODO: Check if original_image exists, raise exception if not
        # TODO: Try to apply processor.process() to original_image
        # TODO: Store result in self.processed_image
        # TODO: Add copy of processed_image to image_history
        # TODO: Return True on success
        # TODO: Handle exceptions and raise with descriptive message
        pass
    
    def reset_image(self):
        """Reset image to original"""
        # TODO: Check if original_image exists
        # TODO: Copy original_image to processed_image
        # TODO: Return True if successful, False otherwise
        pass
    
    def save_image(self, file_path):
        """Save the processed image"""
        # TODO: Check if processed_image exists, raise exception if not
        # TODO: Try to save processed_image to file_path
        # TODO: Return True on success
        # TODO: Handle exceptions and raise with descriptive message
        pass
    
    def get_image_info(self):
        """Get image information"""
        # TODO: Check if original_image exists
        # TODO: Return dictionary with:
        #      - filename (use os.path.basename)
        #      - size (tuple of width, height)
        #      - mode (image color mode)
        #      - format (image format)
        # TODO: Return None if no image loaded
        pass
    
    def undo(self):
        """Undo last operation"""
        # TODO: Check if image_history has more than 1 item
        # TODO: Remove last item from image_history
        # TODO: Set processed_image to copy of new last item
        # TODO: Return True if successful, False otherwise
        pass
