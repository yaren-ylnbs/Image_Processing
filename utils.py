# ======================== utils.py ========================
"""
Utility functions and helper classes
"""
import os
from typing import Tuple, Optional
from PIL import Image

class ImageValidator:
    """Validates image files and operations"""
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    MAX_FILE_SIZE_MB = 50
    
    @staticmethod
    def is_valid_image_file(file_path: str) -> bool:
        """Check if file is a valid image"""
        #Check if file exists
        #Check if file extension is in SUPPORTED_EXTENSIONS
        #Check if file size is within MAX_FILE_SIZE_MB
        #Try to open and verify image with PIL
        #Return True if valid, False otherwise
        # 1. Dosya var mı kontrol et
        if not os.path.exists(file_path):
            return False
            
        # 2. Dosya uzantısını kontrol et (küçük harfe çevirerek)
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in ImageValidator.SUPPORTED_EXTENSIONS:
            return False
            
        # 3. Dosya boyutunu kontrol et (Byte -> MB çevirimi)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > ImageValidator.MAX_FILE_SIZE_MB:
            return False
            
        # 4. PIL ile dosyayı açmayı dene ve doğrula (verify)
        try:
            with Image.open(file_path) as img:
                img.verify() # Dosyanın içeriğini okumadan header kontrolü yapar (hızlıdır)
            return True
        except Exception:
            return False
    
    @staticmethod
    def validate_dimensions(image: Image.Image, 
                           max_width: int = 10000, 
                           max_height: int = 10000) -> bool:
        """Validate image dimensions"""
        #Get image dimensions
        #Check if dimensions are within max limits
        #Return validation result
        width, height = image.size
        if width > max_width or height > max_height:
            return False
        return True
    
class ImageResizer:
    """Handles image resizing operations"""
    
    @staticmethod
    def resize_to_fit(image: Image.Image, 
                     max_width: int, 
                     max_height: int) -> Image.Image:
        """Resize image to fit within max dimensions"""
        #Get current image dimensions
        #Calculate width and height ratios
        #Choose minimum ratio (max 1.0)
        #If ratio < 1, resize image with LANCZOS resampling
        #Return resized or original image
        img_width, img_height = image.size
        width_ratio = max_width / img_width
        height_ratio = max_height / img_height
        ratio = min(width_ratio, height_ratio, 1.0)
        if ratio < 1.0:
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            return image.resize((new_width, new_height), Image.LANCZOS)
        return image
    
    @staticmethod
    def calculate_aspect_ratio(width: int, height: int) -> float:
        """Calculate aspect ratio"""
        #Calculate and return width/height ratio
        #Handle division by zero
        return width / height if height != 0 else 0.0


class FileManager:
    """Handles file operations"""
    
    @staticmethod
    def get_file_info(file_path: str) -> dict:
        """Get file information"""
        #Check if file exists
        #Get file statistics using os.stat
        #Return dictionary with:
        #      - filename
        #      - path
        #      - size_bytes
        #      - size_mb
        #      - extension
        #Return None if file doesn't exist
        if not os.path.exists(file_path):
            return None
        stats = os.stat(file_path)
        return {
            'filename': os.path.basename(file_path),
            'path': file_path,
            'size_bytes': stats.st_size,
            'size_mb': stats.st_size / (1024 * 1024),
            'extension': os.path.splitext(file_path)[1].lower()
        }
        

    
    @staticmethod
    def ensure_directory_exists(file_path: str) -> bool:
        """Ensure directory exists for file path"""
        #Get directory from file path
        #Check if directory exists
        #Create directory if it doesn't exist
        #Return True if successful, False otherwise
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                return True
            except Exception:
                return False
        return True
    
