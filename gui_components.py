# ======================== gui_components.py ========================
"""
GUI component classes
"""
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class MenuManager:
    """Manages application menu bar"""
    
    def __init__(self, root, callback_handler):
        self.root = root
        self.callback = callback_handler
        self.menubar = None
        
    def create_menu(self):
        """Create menu bar"""
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)
        
        self._create_file_menu()
        self._create_filter_menu()
        self._create_adjustment_menu()
        self._create_help_menu()
        
        return self.menubar
    
    def _create_file_menu(self):
        """Create file menu"""
        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.callback.browse_image)
        file_menu.add_command(label="Save", command=self.callback.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
    
    def _create_filter_menu(self):
        """Create filter menu"""
        filter_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Filters", menu=filter_menu)
        
        # Basic filters
        basic_filters = tk.Menu(filter_menu, tearoff=0)
        filter_menu.add_cascade(label="Basic Filters", menu=basic_filters)
        basic_filters.add_command(label="Blur", command=lambda: self.callback.apply_filter("blur"))
        basic_filters.add_command(label="Sharpen", command=lambda: self.callback.apply_filter("sharpen"))
        basic_filters.add_command(label="Edge Detect", command=lambda: self.callback.apply_filter("edge_detect"))
        basic_filters.add_command(label="Emboss", command=lambda: self.callback.apply_filter("emboss"))
        
        # Color filters
        color_filters = tk.Menu(filter_menu, tearoff=0)
        filter_menu.add_cascade(label="Color Filters", menu=color_filters)
        color_filters.add_command(label="Grayscale", command=lambda: self.callback.apply_filter("grayscale"))
        color_filters.add_command(label="Sepia", command=lambda: self.callback.apply_filter("sepia"))
        color_filters.add_command(label="Invert", command=lambda: self.callback.apply_filter("invert"))
        color_filters.add_command(label="Solarize", command=lambda: self.callback.apply_filter("solarize"))
        
        # Advanced filters
        advanced_filters = tk.Menu(filter_menu, tearoff=0)
        filter_menu.add_cascade(label="Advanced Filters", menu=advanced_filters)
        advanced_filters.add_command(label="Gaussian Blur", command=lambda: self.callback.apply_filter("gaussian_blur"))
        advanced_filters.add_command(label="Motion Blur", command=lambda: self.callback.apply_filter("motion_blur"))
        advanced_filters.add_command(label="Canny Edge", command=lambda: self.callback.apply_filter("canny_edge"))
        advanced_filters.add_command(label="Median Filter", command=lambda: self.callback.apply_filter("median_filter"))

        advanced_filters.add_separator()
        advanced_filters.add_command(label = "HOG Visualization", command = lambda: self.callback.apply_filter("hog"))

        advanced_filters.add_command(label ="Person Detection", command = lambda: self.callback.apply_filter("person detect"))
        advanced_filters.add_command(label = "Custom Detection (horse)", command = lambda: self.callback.apply_filter("custom_detect"))
    
    def _create_adjustment_menu(self):
        """Create adjustment menu"""
        adjust_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Adjustments", menu=adjust_menu)
        adjust_menu.add_command(label="Increase Brightness", command=lambda: self.callback.apply_enhancement("brightness_up"))
        adjust_menu.add_command(label="Decrease Brightness", command=lambda: self.callback.apply_enhancement("brightness_down"))
        adjust_menu.add_command(label="Increase Contrast", command=lambda: self.callback.apply_enhancement("contrast_up"))
        adjust_menu.add_command(label="Decrease Contrast", command=lambda: self.callback.apply_enhancement("contrast_down"))
    
    def _create_help_menu(self):
        """Create help menu"""
        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.callback.show_about)

class ImageDisplay:
    """Handles image display on canvas"""
    
    def __init__(self, canvas, title="Image"):
        self.canvas = canvas
        self.title = title
        self.photo_image = None
        
    def display_image(self, image):
        """Display image on canvas"""
        if image is None:
            return
            
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 500
        if canvas_height <= 1:
            canvas_height = 400
            
        display_image = self._resize_for_display(image, canvas_width-20, canvas_height-20)
        
        self.canvas.delete("all")
        
        self.photo_image = ImageTk.PhotoImage(display_image)
        x = canvas_width // 2
        y = canvas_height // 2
        self.canvas.create_image(x, y, anchor=tk.CENTER, image=self.photo_image)
    
    def _resize_for_display(self, image, max_width, max_height):
        """Resize image for display"""
        img_width, img_height = image.size
        
        width_ratio = max_width / img_width
        height_ratio = max_height / img_height
        ratio = min(width_ratio, height_ratio, 1.0)
        
        if ratio < 1.0:
            new_width = int(img_width * ratio)
            new_height = int(img_height * ratio)
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def clear_display(self):
        """Clear canvas"""
        self.canvas.delete("all")

class StatusManager:
    """Manages status bar"""
    
    def __init__(self, status_var):
        self.status_var = status_var
    
    def set_status(self, message):
        """Update status message"""
        self.status_var.set(message)
    
    def set_ready(self):
        """Set ready status"""
        self.status_var.set("Ready - Select an image...")
    
    def set_processing(self, operation):
        """Set processing status"""
        self.status_var.set(f"Processing: {operation}...")
    
    def set_error(self, error):
        """Set error status"""
        self.status_var.set(f"Error: {error}")

