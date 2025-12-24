# ======================== enhancements.py ========================
"""
Image enhancement implementations
"""
from PIL import ImageEnhance
from base_classes import ImageEnhancement

class BrightnessEnhancement(ImageEnhancement):
    """Adjust image brightness"""
    def __init__(self, factor=1.3):
        #Initialize brightness enhancement
        super().__init__("Brightness", factor)

    def process(self, image):
        #Implement brightness adjustment using ImageEnhance.Brightness
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(self.factor)
    

class ContrastEnhancement(ImageEnhancement):
    """Adjust image contrast"""
    def __init__(self, factor=1.3):
        #Initialize contrast enhancement
        super().__init__("Contrast", factor)
    
    def process(self, image):
        #Implement contrast adjustment using ImageEnhance.Contrast
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(self.factor)

class ColorEnhancement(ImageEnhancement):
    """Adjust color saturation"""
    def __init__(self, factor=1.3):
        #Initialize color enhancement
        super().__init__("Color", factor)
    
    def process(self, image):
        #Implement color saturation adjustment using ImageEnhance.Color
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(self.factor)

class SharpnessEnhancement(ImageEnhancement):
    """Adjust image sharpness"""
    def __init__(self, factor=1.3):
        #Initialize sharpness enhancement
        super().__init__("Sharpness", factor)
    
    def process(self, image):
        #Implement sharpness adjustment using ImageEnhance.Sharpness
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(self.factor)
