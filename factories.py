# ======================== factories.py ========================
"""
Factory classes for creating filters and enhancements
"""
from filters import *
from enhancements import *

class FilterFactory:
    """Factory for creating filter instances"""
    
    def __init__(self):
        #Create a dictionary mapping filter names to filter classes
            self.filters = {
            "blur": BlurFilter,
            "grayscale": GrayscaleFilter,
            "hog": HOGVisualizationFilter,
            "person detect": PersonDetectionFilter,
            "custom_detect" : CustomObjectDetectionFilter,
            "sharpen": SharpenFilter,
            "edge_detect": EdgeDetectionFilter,
            "emboss": EmbossFilter,
            "sepia": SepiaFilter,
            "invert": InvertFilter,
            "solarize": SolarizeFilter,
            "gaussian_blur": GaussianBlurFilter,    
            "motion_blur": MotionBlurFilter,
            "canny_edge": CannyEdgeFilter,
            "median_filter": MedianFilter,
          
        }
    
    def create_filter(self, filter_name):
        """Create a filter instance by name"""
        # Get the filter class from self.filters dictionary
        #Create and return an instance of the filter class
        # Raise ValueError if filter_name not found
        
        filter_class = self.filters.get(filter_name.lower())
        if filter_class is None:
            raise ValueError(f"Unknown filter: {filter_name}")
        return filter_class()
    
    def get_available_filters(self):
        """Get list of available filter names"""
        #Return list of filter names from self.filters dictionary
        return list(self.filters.keys())

class EnhancementFactory:
    """Factory for creating enhancement instances"""
    
    def __init__(self):
        #Create a dictionary mapping enhancement names to enhancement creation functions
        # Example: self.enhancements = {'brightness_up': lambda: BrightnessEnhancement(1.3), ...}
        self.enhancements = {
            "brightness_up": lambda: BrightnessEnhancement(1.2),   # Parlaklığı %20 artır
            "brightness_down": lambda: BrightnessEnhancement(0.8), # Parlaklığı %20 azalt
            "contrast_up": lambda: ContrastEnhancement(1.2),       # Kontrastı artır
            "contrast_down": lambda: ContrastEnhancement(0.8),     # Kontrastı azalt
            "sharpness_up": lambda: SharpnessEnhancement(2.0),     # Keskinliği artır
            "sharpness_down": lambda: SharpnessEnhancement(0.5),   # Keskinliği azalt (Bulanıklaştır)
            "color_up": lambda: ColorEnhancement(1.2),             # Renk doygunluğunu artır
            "color_down": lambda: ColorEnhancement(0.8)            # Renk doygunluğunu azalt (Siyah beyaza yaklaş)
        }

    
    def create_enhancement(self, enhancement_name):
        """Create an enhancement instance by name"""
        #Get the enhancement creation function from self.enhancements dictionary
        #Call the function and return the enhancement instance
        #Raise ValueError if enhancement_name not found
        enhancement_creator = self.enhancements.get(enhancement_name.lower())

        if enhancement_name in self.enhancements:
            # Sözlükten fonksiyonu al ve () ile çalıştırarak nesneyi üret
            return self.enhancements[enhancement_name]()
        else:
            raise ValueError(f"Enhancement '{enhancement_name}' not found.")
            
    def get_available_enhancements(self):
        """Get list of available enhancement names"""
        #Return list of enhancement names from self.enhancements dictionary
        return list(self.enhancements.keys())
