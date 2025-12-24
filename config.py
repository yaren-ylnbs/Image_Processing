# ======================== config.py ========================
"""
Configuration settings for the application
"""

class AppConfig:
    """Application configuration"""
    
    # Window settings
    WINDOW_TITLE = "OOP Image Processing Application"
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 800
    WINDOW_BG = '#f0f0f0'
    
    # Canvas settings
    CANVAS_WIDTH = 500
    CANVAS_HEIGHT = 400
    CANVAS_BG = 'white'
    CANVAS_BORDER = 2
    
    # Image settings
    MAX_DISPLAY_WIDTH = 480
    MAX_DISPLAY_HEIGHT = 380
    DEFAULT_IMAGE_FORMAT = 'PNG'
    
    # Filter settings
    GAUSSIAN_BLUR_RADIUS = 2
    MEDIAN_FILTER_KERNEL = 5
    SOLARIZE_THRESHOLD = 128
    
    # Enhancement settings
    BRIGHTNESS_INCREASE_FACTOR = 1.3
    BRIGHTNESS_DECREASE_FACTOR = 0.7
    CONTRAST_INCREASE_FACTOR = 1.3
    CONTRAST_DECREASE_FACTOR = 0.7
    COLOR_INCREASE_FACTOR = 1.3
    COLOR_DECREASE_FACTOR = 0.7
    SHARPNESS_INCREASE_FACTOR = 1.5
    SHARPNESS_DECREASE_FACTOR = 0.5
    
    # File settings
    SUPPORTED_FORMATS = [
        ("All Image Formats", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
        ("JPEG", "*.jpg *.jpeg"),
        ("PNG", "*.png"),
        ("BMP", "*.bmp"),
        ("TIFF", "*.tiff"),
        ("GIF", "*.gif")
    ]
    
    SAVE_FORMATS = [
        ("PNG", "*.png"),
        ("JPEG", "*.jpg"),
        ("BMP", "*.bmp"),
        ("TIFF", "*.tiff")
    ]
    
    # UI Text
    TITLE_TEXT = "OOP Image Processing Application"
    ORIGINAL_LABEL = "Original Image"
    PROCESSED_LABEL = "Processed Image"
    QUICK_FILTERS_LABEL = "Quick Filters"
    
    # Status messages
    STATUS_READY = "Ready - Select an image..."
    STATUS_NO_FILE = "No file selected"
    STATUS_PROCESSING = "Processing: {}..."
    STATUS_IMAGE_LOADED = "Image loaded: {} ({}x{})"
    STATUS_FILTER_APPLIED = "{} filter applied"
    STATUS_IMAGE_SAVED = "Image saved: {}"
    STATUS_IMAGE_RESET = "Image reset"
    STATUS_ERROR = "Error: {}"
