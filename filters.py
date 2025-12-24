# ======================== filters.py ========================
"""
Concrete filter implementations
"""
import os
from PIL import Image, ImageFilter, ImageOps
import cv2
import numpy as np
from base_classes import Filter
from hog_implementation import visualize_hog_pil
from hog_implementation import compute_hog_descriptor
import pickle

class BlurFilter(Filter):
    """Apply blur filter"""
    def __init__(self):
        super().__init__("Blur")
    
    def process(self, image):
        """ Apply blur filter using ImageFilter.BLUR """
        return image.filter(ImageFilter.BLUR)

class SharpenFilter(Filter):
    """Apply sharpen filter"""
    def __init__(self):
        super().__init__("Sharpen")
    
    def process(self, image):
        """ Apply sharpen filter using ImageFilter.SHARPEN """
        return image.filter(ImageFilter.SHARPEN)

class EdgeDetectionFilter(Filter):
    """Apply edge detection filter"""
    def __init__(self):
        super().__init__("Edge Detection")
    
    def process(self, image):
        """ Apply edge detection filter using ImageFilter.FIND_EDGES """
        return image.filter(ImageFilter.FIND_EDGES)

class EmbossFilter(Filter):
    """Apply emboss filter"""
    def __init__(self):
        super().__init__("Emboss")
    
    def process(self, image):
       """ Apply emboss filter using ImageFilter.EMBOSS """
       return image.filter(ImageFilter.EMBOSS)

class GrayscaleFilter(Filter):
    """Convert to grayscale"""
    def __init__(self):
        super().__init__("Grayscale")
    
    def process(self, image):
        """Convert image to grayscale using ImageOps.grayscale"""
        """Convert back to RGB mode after grayscale conversion"""
        gray = ImageOps.grayscale(image)
        return gray.convert("RGB")  

class SepiaFilter(Filter):
    """Apply sepia tone filter"""
    def __init__(self):
        super().__init__("Sepia")
    
    def process(self, image):
        """Convert image to numpy array"""

        """ Apply sepia matrix transformation
        Sepia matrix:
        [[0.393, 0.769, 0.189, 0],
         [0.349, 0.686, 0.168, 0],
         [0.272, 0.534, 0.131, 0],
         [0, 0, 0, 1]]
        Clip values to 0-255 range
        Convert back to PIL Image"""
    
        image_np = np.array(image)
        sepia_matrix = np.array([
                [0.393, 0.769, 0.189, 0],
            [0.349, 0.686, 0.168, 0],
            [0.272, 0.534, 0.131, 0],])
        sepia_img = cv2.transform(image_np,sepia_matrix)

        # 4. Değerleri 0-255 arasına sıkıştır (Clip)
        # 255'i geçen değerler bozulma yapmasın diye 255'e sabitlenir
        sepia_img = np.clip(sepia_img, 0, 255)

        # 5. Veri tipini uint8 (resim formatı) yap
        sepia_img = sepia_img.astype(np.uint8)
        
        # 6. Tekrar PIL Image formatına çevirip döndür
        return Image.fromarray(sepia_img)

class InvertFilter(Filter):
    """Invert image colors"""
    def __init__(self):
        super().__init__("Invert")
    
    def process(self, image):
        #Invert image colors using ImageOps.invert
        return ImageOps.invert(image)

class SolarizeFilter(Filter):
    """Apply solarize effect"""
    def __init__(self):
        super().__init__("Solarize")
    
    def process(self, image):
        #Apply solarize effect using ImageOps.solarize with threshold=128
        return ImageOps.solarize(image, threshold = 128)

class GaussianBlurFilter(Filter):
    """Apply Gaussian blur"""
    def __init__(self, radius=2):
        super().__init__("Gaussian Blur")
        self.radius = radius
    
    def process(self, image):
        #Apply Gaussian blur using ImageFilter.GaussianBlur with self.radius
        return image.filter(ImageFilter.GaussianBlur(self.radius))

class CannyEdgeFilter(Filter):
    """Apply Canny edge detection"""
    def __init__(self):
        super().__init__("Canny Edge")
    
    def process(self, image):
        #Convert image to numpy array
        #Convert to grayscale using cv2.cvtColor
        #Apply Canny edge detection using cv2.Canny with thresholds 100, 200
        #Convert edges back to RGB
        #Convert back to PIL Image
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        rgb_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb_edges)

class MedianFilter(Filter):
    """Apply median filter for noise reduction"""
    def __init__(self, kernel_size=5):
        super().__init__("Median Filter")
        self.kernel_size = kernel_size
    
    def process(self, image):
        #Convert image to numpy array
        #Apply median blur using cv2.medianBlur with self.kernel_size
        #Convert back to PIL Image
        image_np = np.array(image)
        blured = cv2.medianBlur(image_np, self.kernel_size)
        return Image.fromarray(blured)

class MotionBlurFilter(Filter):
    """Apply motion blur effect"""
    def __init__(self):
        super().__init__("Motion Blur")
    
    def process(self, image):
        #Create a 9x9 motion blur kernel
        # Hint: Use diagonal 1s in the kernel matrix, scale by 9
        #Apply kernel using ImageFilter.Kernel
        blur_kernel = np.zeros((9,9))
        np.fill.diagonal(blur_kernel, 1)
        blur_kernel = blur_kernel / 9.0
        return image.filter(ImageFilter.Kernel((9,9), blur_kernel.flatten()))
    

class HOGVisualizationFilter(Filter):
    def __init__(self):
        super().__init__("HOG Visualization")

    def process(self, image):
        # image: GUI'den gelen PIL.Image
        return visualize_hog_pil(image)

class PersonDetectionFilter(Filter):
    """OpenCV'nin HOG+SVM pretrained modelini kullanarak insan tespiti"""
    def __init__(self):
        super().__init__("Person Detection")
        self.hog = cv2.HOGDescriptor() #opencv'nin hazır hog tanımlayıcısını başlat
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) #insan tespiti için svm detektörünü yükle
        self.conf_thresh = 0.35  # başlangıç için;
        self.last_num_detections = 0 # son çalıştırmada bulunan kişi sayısı


    def process(self,image):
        img_np = np.array(image) #PIL görüntüsünü np array çevr
        #analizi hızlandırmak için resmi geçici olarak küçült
        height, width = img_np.shape[:2]
        max_width = 800

        scale_factor = 1.0
        if width > max_width:
            scale_factor = max_width / width
            new_width = max_width
            new_height = int(height*scale_factor)
            img_small = cv2.resize(img_np, (new_width, new_height))
        else:
            img_small = img_np
            
        #renkli görüntü üzerinde tespit yapma
        rects, weights = self.hog.detectMultiScale(img_small, winStride=(4,4), padding=(4,4), scale=1.05) #winStride: pencere kaydırma adımı padding: görüntü kenarlarına eklenecek boşluk scale: görüntü piramidi(perspektif) ölçek faktörü
        confidences = []
        for w in weights:
            if hasattr(w,"__len__"):
                confidences.append(float(w[0]))
            else:
                confidences.append(float(w))
            
        confidences = np.array(confidences)

        img_out = img_np.copy() #sonuçlar için kopya al

        if len(rects) > 0: #insan bulunduysa
            #koordinatları orijinal boyutuna çevir
            rects = np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
            rects = rects / scale_factor
            rects = rects.astype("int") #tamsayıya çevir
        else:
        # Hiç tespit yoksa rects'i boş bir (0,4) array yapalım
            rects = np.empty((0, 4), dtype=int)

        #nms ( non maximum suppression) - çakışan kutuları temizleme -- matematiksel olarak kutuların kesişim alanını hesaplayarak üst üste binen kutulardan eleme yapıyor.
        pick = self.non_max_suppression_fast(rects, overlapThresh = 0.65)

        num_persons = 0


        for (xA, yA, xB, yB) in pick: #sadece seçilen kutuları çiz
            idx_matches = np.where(
                (rects[:,0] == xA) &
                (rects[:,1] == yA) &
                (rects[:,2] == xB) &
                (rects[:,3] == yB)
            )[0]
            
            score  = None
            if len(idx_matches) > 0:
                idx=idx_matches[0]
                score = float(confidences[idx])
                
            if score is not None and score < self.conf_thresh:
                continue  #eşik değerinin (0.35 belirledik) altındaysa kutuyu hiç çizme

            num_persons += 1
            if score is not None:
                label = f"Person: {score:.2f}"
            else:
                label = "Person"

            cv2.rectangle(img_out, (xA, yA), (xB, yB), (0, 255, 0), 3) #yeşil kutuyu çiz
            cv2.putText(img_out,label, (xA, yA-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
         #Kişi sayısını threshold sonrası kaydet ve yazdır
        self.last_num_detections = num_persons
        print(f"Bu görüntüde tespit edilen kişi sayısı: {num_persons}")

        return Image.fromarray(img_out) 

    def non_max_suppression_fast(self, boxes, overlapThresh):  #çakışan kutuları temizleyen fonksiyon-- iki kutu %65'ten fazla örtüşüyorsa sadece birini tut.
        if len(boxes) == 0:
            return []
        
        if boxes.dtype.kind == "i": #bölme işlemi için floata çeviriyoruz
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2-x1+1) * (y2-y1+1) #kutuların alanları
        idxs = np.argsort(y2) #y koordinatına göre sırala.


        while len(idxs) > 0 :
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            #kesişim alanını bul
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")           
        
    import pickle  # Modeli yüklemek için
       
class CustomObjectDetectionFilter(Filter):
    """
    PDF Madde 4.2 & 5: Custom Object Detection + Multi-Scale
    - Eğitilen modeli yükler.
    - Resmi piramit mantığıyla küçülterek (Multi-scale) tarar.
    - Böylece at resimdeki boyutu ne olursa olsun yakalanır.
    """
    def __init__(self):
        super().__init__("Custom Object Detection")
        
        self.model_path = "trained_classifier.pkl"
        self.window_size = (64, 128) # Eğitim boyutuyla AYNI olmalı
        self.step_size = 4          # Tarama adımı
        
        self.model = None #değişkeni boş oluştur
        self.load_model() #model yükleme fonk çağır
    
    def load_model(self):
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model Yüklendi: {self.model_path}")
        except Exception as e:
            print(f"Model Hatası: {e}")
            self.model = None

    def process(self, image):
        if self.model is None: return image

        img_pil = image.copy()
        img_np = np.array(img_pil)
        
        # Renkli çıktı için
        if len(img_np.shape) == 2:
            img_out = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            img_gray = img_np
        else:
            img_out = img_np.copy()
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        detections = [] 
        winW, winH = self.window_size

        curr_img = img_gray.copy()
        current_scale = 1.0
        
        # Sonsuz döngüden kaçınmak için güvenlik
        while True:
            h, w = curr_img.shape[:2]
            
            # Eğer resim pencereden küçükse dur
            if h < winH or w < winW:
                break

# --- SLIDING WINDOW (Mevcut kodunuzun aynısı, sadece step_size=4 ile) ---
            for y in range(0, h - winH, self.step_size):
                for x in range(0, w - winW, self.step_size):
                    window = curr_img[y:y+winH, x:x+winW]
                    
                    features = compute_hog_descriptor(window, cell_size=(8,8), block_size=(2,2), num_bins=9)
                    
                    if len(features) > 0:
                        score = self.model.decision_function([features])[0]
                        
                        # Threshold'u çok düşürmeden makul bir seviyede tutun
                        if score > 0.35: # 0.0, karar sınırıdır.
                            # Koordinatları orijinal boyuta çevir
                            real_x = int(x * current_scale)
                            real_y = int(y * current_scale)
                            real_w = int(winW * current_scale)
                            real_h = int(winH * current_scale)
                            detections.append([real_x, real_y, real_x+real_w, real_y+real_h, score])

            # --- BİR SONRAKİ SCALE İÇİN KÜÇÜLTME ---
            # Resmi her turda %10 küçültüyoruz (Daha hassas arama için %5 de yapılabilir)
            curr_img = cv2.resize(curr_img, (0,0), fx=0.9, fy=0.9)
            current_scale = current_scale / 0.9 # Koordinatları geri çarpmak için scale'i büyütüyoruz
        # --- NMS (Temizlik) ---
        if len(detections) > 0:
            detections = np.array(detections)
            boxes = detections[:, :4].astype(int)
            scores = detections[:, 4]

            pick = self.non_max_suppression_fast(boxes, overlapThresh=0.3)
            print(f"NMS Sonrası Tespit: {len(pick)}")

            for (xA, yA, xB, yB) in pick:
                # Bu kutunun boxes içindeki index'ini bul
                idx_matches = np.where(
                    (boxes[:, 0] == xA) &
                    (boxes[:, 1] == yA) &
                    (boxes[:, 2] == xB) &
                    (boxes[:, 3] == yB)
                )[0]
                
                score = None
                if len(idx_matches) > 0:
                    idx = idx_matches[0]
                    score = float(scores[idx])

                if score is not None:
                    label = f"Horse: {score:.2f}"
                else:
                    label = "Horse"

                cv2.rectangle(img_out, (xA, yA), (xB, yB), (255, 0, 0), 3)
                cv2.putText(img_out, label, (xA+5, yA+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            print("Hiç at bulunamadı.")

        return Image.fromarray(img_out)

    def non_max_suppression_fast(self, boxes, overlapThresh):
        if len(boxes) == 0: return []
        if boxes.dtype.kind == "i": boxes = boxes.astype("float")
        pick = []
        x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        return boxes[pick].astype("int")