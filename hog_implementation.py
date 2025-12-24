import numpy as np
import cv2

def compute_gradients(image: np.ndarray):  
    #görüntünün x ve y yönündeki gradyanlarını hesaplar. girdi: gri tonlamalı görüntü(np array) çıktı: magniture(büyüklük), angle(açı)

    img = image.astype(np.float32)

    Gy, Gx = np.gradient(img) #np.gradient (Gy,Gx) döner.

    magnitude = np.sqrt(Gx**2 + Gy**2) #formüle göre büyüklük
    angle = np.rad2deg(np.arctan2(Gy,Gx)) #formüle göre açı

    angle[angle <0] += 180.0  #açının 0-180 aralığında olması istenmiş.

    return magnitude,angle

def create_cell_histogram(cell_magnitude: np.ndarray,cell_angle: np.ndarray,num_bins: int = 9) -> np.ndarray:
    """
    yönelim histogramı oluşturulacak. girdi: hücre gradyan büyüklükleri, açıları,bin sayıları çıktı: histogram 
    0-180 aralığı eşit binlere bölünecek.
    """

    hist= np.zeros(num_bins, dtype = np.float32)
    bin_width = 180.0/num_bins #0-180 derece eşit aralıklara bölünüyor.

    angles = cell_angle.flatten()
    mags = cell_magnitude.flatten()

    for ang, mag in zip(angles,mags):
        bin_idx = int(ang // bin_width) #hangi bine düşeceğini buluyor.
        if bin_idx >= num_bins:
            bin_idx = num_bins - 1
        hist[bin_idx] += mag

    return hist

def normalize_block(block_histogram: np.ndarray, method: str = "L2", eps: float = 1e-5) -> np.ndarray:
    """
    blok histogramının normalizasyonu. girdi: blok histogtamı, normalizasyon methodu  çıktı: normalize edilmiş histogram

    L2 norm: h/sqrt(||h||**2 + eps**2)
    """
    h = block_histogram.astype(np.float32)

    if method == "L2":
        norm = np.sqrt(np.sum(h**2) + eps**2)
        return h/norm
        
def compute_hog_descriptor(image: np.ndarray, cell_size= (8,8), block_size = (2,2), num_bins: int = 9) -> np.ndarray:
    """
    tam HOG descriptor hesaplar. girdi: gri görüntü, hücre boyutu, blok boyutu, bin sayısı çıktı: HOG özellik vektörü (1d numpy array)

    *sliding window ile bloklar taranacak, her blok için histogramlar normalize edilecektir.
    """

    magnitude, angle = compute_gradients(image) #gradyanların hesaplanması

    h,w = image.shape
    cell_h, cell_w = cell_size

    n_cells_y = h // cell_h #hücre sayısı
    n_cells_x = w // cell_w

    cell_histograms = np.zeros((n_cells_y, n_cells_x, num_bins), dtype = np.float32)

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            y_start = i * cell_h
            y_end = y_start + cell_h
            x_start = j * cell_w
            x_end = x_start + cell_w

            cell_mag = magnitude[y_start:y_end, x_start:x_end]
            cell_ang = angle[y_start:y_end, x_start:x_end]

            hist = create_cell_histogram(cell_mag, cell_ang, num_bins = num_bins)
            cell_histograms[i, j, :] = hist
    
    bh,bw = block_size #bloklara böl ve normalize et

    n_blocks_y = n_cells_y - bh + 1 #sliding window, stride = 1
    n_blocks_x = n_cells_x - bw + 1

    hog_features = []

    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = cell_histograms[by: by+bh, bx:bx+bw, :]
            block_vec = block.ravel()
            block_norm = normalize_block(block_vec, method="L2")
            hog_features.append(block_norm)

    if len(hog_features) == 0:
        return np.array([], dtype=np.float32)

    # 5) Tüm blok vektörlerini birleştir
    hog_vector = np.concatenate(hog_features, axis=0)
    return hog_vector


def visualize_hog(image: np.ndarray,
                  cell_size=(12, 12),
                  num_bins: int = 10) -> np.ndarray:
    """
    HOG özelliklerini görselleştirir. 
    Girdi : Gri tonlamalı görüntü
    Çıktı : HOG görselleştirmesini içeren tek-kanallı (uint8) görüntü (numpy array)

    """
        # 1) Gradyanları hesapla
    magnitude, angle = compute_gradients(image)

    h, w = image.shape
    cell_h, cell_w = cell_size
    n_cells_y = h // cell_h
    n_cells_x = w // cell_w

    # 2) Her hücre için histogram
    cell_histograms = np.zeros((n_cells_y, n_cells_x, num_bins), dtype=np.float32)

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            y_start = i * cell_h
            y_end = y_start + cell_h
            x_start = j * cell_w
            x_end = x_start + cell_w

            cell_mag = magnitude[y_start:y_end, x_start:x_end]
            cell_ang = angle[y_start:y_end, x_start:x_end]

            cell_histograms[i, j, :] = create_cell_histogram(
                cell_mag, cell_ang, num_bins=num_bins
            )

    # 3) Çizim için boş görüntü
    hog_image = np.zeros((h, w), dtype=np.uint8)
    bin_width = 180.0 / float(num_bins)

    max_magnitude_global = cell_histograms.max() #tüm resimlerdeki en güçlü gradyanı buluyoruz ki ona göre oranlanabilsin

    # 4) Her hücrenin merkezine histogram çizgileri çiz
    for i in range(n_cells_y):
        for j in range(n_cells_x):
            hist = cell_histograms[i, j, :]

            if hist.max() < 0.10 * max_magnitude_global:   #bir hücredeki en üüyük değer, global maksimumun %5'inden küçükse çizme örn gökyüzü
                continue
            cell_max = hist.max()

            # Hist değerlerini 0–1 aralığına getir (sadece görsel için)
            if cell_max > 0.0:
                hist = hist / cell_max

            # Hücre merkez koordinatı
            cy = int(i * cell_h + cell_h / 2.0)
            cx = int(j * cell_w + cell_w / 2.0)

            for b in range(num_bins):
                # Açı (derece → radyan)
                angle_deg = float(b * bin_width + bin_width / 2.0)
                angle_rad = float(np.deg2rad(angle_deg))

                # Çizgi uzunluğu (histogram değeri ile orantılı)
                length = float(hist[b]) * (min(cell_h, cell_w) / 2.0)

                dx = length * np.cos(angle_rad)
                dy = length * np.sin(angle_rad)

                x1 = int(cx - dx)
                y1 = int(cy - dy)
                x2 = int(cx + dx)
                y2 = int(cy + dy)

                cv2.line(hog_image, (x1, y1), (x2, y2), 255, 1)

    return hog_image


from PIL import Image

def visualize_hog_pil(pil_image, cell_size=(12, 12), num_bins=10):
    """
    GUI'den gelen PIL.Image için HOG görselleştirme yapan helper.
    1) PIL → gri numpy
    2) visualize_hog (numpy) çağır
    3) Sonucu tekrar PIL.Image'e çevir
    """
    # 1) Griye çevir
    gray = pil_image.convert("L")
    img_np = np.array(gray)

    # >>> HOG vektörünü hesaplayıp boyutunu yazdıralım
    hog_vec = compute_hog_descriptor(
        img_np,
        cell_size=cell_size,
        block_size=(2, 2),
        num_bins=num_bins
    )
    print(f"HOG vektör boyutu (cell_size={cell_size}, num_bins={num_bins}): {hog_vec.size}")

    # 2) HOG görselleştirmesini hesapla
    hog_vis = visualize_hog(img_np, cell_size=cell_size, num_bins=num_bins)

    # 3) Tek kanallı (uint8) numpy → PIL
    hog_pil = Image.fromarray(hog_vis)

    # GUI iki panelde de renkli görüntü beklediği için RGB'ye çeviriyoruz
    return hog_pil.convert("RGB")




