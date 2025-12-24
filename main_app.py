"""
Main application class
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os

from image_manager import ImageManager
from factories import FilterFactory, EnhancementFactory
from gui_components import MenuManager, ImageDisplay, StatusManager

class ImageProcessingApplication:
    """Main application class"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("OOP Image Processing Application")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Core components
        self.image_manager = ImageManager()
        self.filter_factory = FilterFactory()
        self.enhancement_factory = EnhancementFactory()
        
        # GUI components
        self.original_display = None
        self.processed_display = None
        self.status_manager = None
        
        self._setup_styles()
        self._setup_gui()
        self._setup_menu()
        self._setup_keyboard_shortcuts()
        
    def _setup_styles(self):
        """Setup UI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
    def _setup_gui(self):
        """Setup GUI components"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="OOP Image Processing Application", 
                               font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Original image
        left_panel = ttk.LabelFrame(main_frame, text="Original Image", padding="10")
        left_panel.grid(row=1, column=0, padx=(0, 10), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Right panel - Processed image
        right_panel = ttk.LabelFrame(main_frame, text="Processed Image", padding="10")
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self._setup_file_controls(left_panel)
        self._setup_canvases(left_panel, right_panel)
        self._setup_control_panel(main_frame)
        self._setup_status_bar(main_frame)
        
    def _setup_file_controls(self, parent):
        """Setup file controls"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        browse_btn = ttk.Button(button_frame, text="Select Image", command=self.browse_image)
        browse_btn.pack(side=tk.LEFT)
        
        self.file_path_var = tk.StringVar(value="No file selected")
        file_path_label = ttk.Label(button_frame, textvariable=self.file_path_var, 
                                   font=('Helvetica', 9))
        file_path_label.pack(side=tk.LEFT, padx=(10, 0))
        
    def _setup_canvases(self, left_panel, right_panel):
        """Setup canvases"""
        original_canvas = tk.Canvas(left_panel, width=500, height=400, 
                                   bg='white', relief=tk.SUNKEN, borderwidth=2)
        original_canvas.pack(expand=True, fill=tk.BOTH)
        self.original_display = ImageDisplay(original_canvas, "Original")
        
        processed_canvas = tk.Canvas(right_panel, width=500, height=400, 
                                    bg='white', relief=tk.SUNKEN, borderwidth=2)
        processed_canvas.pack(expand=True, fill=tk.BOTH)
        self.processed_display = ImageDisplay(processed_canvas, "Processed")
        
    def _setup_control_panel(self, parent):
        """Setup control panel"""
        control_frame = ttk.Frame(parent)
        control_frame.grid(row=2, column=0, columnspan=2, pady=(20, 0), sticky=(tk.W, tk.E))
        
        quick_filters_frame = ttk.LabelFrame(control_frame, text="Quick Filters", padding="10")
        quick_filters_frame.pack(fill=tk.X, pady=(0, 10))
        
        filter_buttons = [
            ("Blur", "blur"),
            ("Sharpen", "sharpen"),
            ("Grayscale", "grayscale"),
            ("Edge Detect", "edge_detect"),
            ("Sepia", "sepia"),
            ("Invert", "invert")
        ]
        
        for i, (text, filter_name) in enumerate(filter_buttons):
            btn = ttk.Button(quick_filters_frame, text=text, 
                           command=lambda f=filter_name: self.apply_filter(f), width=12)
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
        
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X)
        
        reset_btn = ttk.Button(action_frame, text="Reset", command=self.reset_image)
        reset_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        save_btn = ttk.Button(action_frame, text="Save", command=self.save_image)
        save_btn.pack(side=tk.LEFT)
        
    def _setup_status_bar(self, parent):
        """Setup status bar"""
        self.status_var = tk.StringVar(value="Ready - Select an image...")
        status_bar = ttk.Label(parent, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.status_manager = StatusManager(self.status_var)
        
    def _setup_menu(self):
        """Setup menu"""
        self.menu_manager = MenuManager(self.root, self)
        self.menu_manager.create_menu()
        
    def _setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind('<Control-o>', lambda e: self.browse_image())
        self.root.bind('<Control-s>', lambda e: self.save_image())
        self.root.bind('<Control-q>', lambda e: self.root.quit())
        
    def browse_image(self):
        """Browse and load image"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("All Image Formats", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tiff"),
                ("GIF", "*.gif")
            ]
        )
        
        if file_path:
            try:
                self.image_manager.load_image(file_path)
                filename = os.path.basename(file_path)
                self.file_path_var.set(f"Selected: {filename}")
                
                self.original_display.display_image(self.image_manager.original_image)
                self.processed_display.display_image(self.image_manager.processed_image)
                
                info = self.image_manager.get_image_info()
                self.status_manager.set_status(f"Image loaded: {info['filename']} ({info['size'][0]}x{info['size'][1]})")
                
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.status_manager.set_error("Failed to load image")
    
    def apply_filter(self, filter_name):
        """Apply filter to image"""
        try:
            filter_processor = self.filter_factory.create_filter(filter_name)
            self.status_manager.set_processing(f"{filter_processor.name} filter")
            
            self.image_manager.apply_processor(filter_processor)
            self.processed_display.display_image(self.image_manager.processed_image)
            
            self.status_manager.set_status(f"{filter_processor.name} filter applied")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_manager.set_error(str(e))
    
    def apply_enhancement(self, enhancement_name):
        """Apply enhancement to image"""
        try:
            enhancement_processor = self.enhancement_factory.create_enhancement(enhancement_name)
            self.status_manager.set_processing(f"{enhancement_processor.name} adjustment")
            
            self.image_manager.apply_processor(enhancement_processor)
            self.processed_display.display_image(self.image_manager.processed_image)
            
            self.status_manager.set_status(f"{enhancement_processor.name} adjustment applied")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_manager.set_error(str(e))
    
    def reset_image(self):
        """Reset image to original"""
        if self.image_manager.reset_image():
            self.processed_display.display_image(self.image_manager.processed_image)
            self.status_manager.set_status("Image reset")
    
    def save_image(self):
        """Save processed image"""
        if self.image_manager.processed_image is None:
            messagebox.showwarning("Warning", "No image to save!")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tiff")
            ]
        )
        
        if file_path:
            try:
                self.image_manager.save_image(file_path)
                messagebox.showinfo("Success", f"Image saved:\n{file_path}")
                self.status_manager.set_status(f"Image saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
OOP Image Processing Application v2.0

Developed with modern OOP principles.

Features:
• Modular class structure
• Factory pattern usage
• Various image filters
• Enhancement algorithms
• Extensible architecture

Developer: Python & Tkinter (OOP)
Libraries: PIL, OpenCV, NumPy
        """
        messagebox.showinfo("About", about_text)

