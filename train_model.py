import os
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import LinearSVC
from base_classes import Filter
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from hog_implementation import compute_hog_descriptor

# Klasör Yolları
POS_PATH = "data/training_set/positive"
NEG_PATH = "data/training_set/negative"
MODEL_PATH = "trained_classifier.pkl" 
RESULTS_PATH = "data/results"

# Klasör yoksa oluştur
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

SIZE = (64, 128) # Eğitim boyutu 

def load_data():
    data = []
    labels = []
    
    # 1. Pozitif (Nesne Var) Resimleri Yükle
    print(f"Pozitif resimler okunuyor...")
    if not os.path.exists(POS_PATH):
        print(f"HATA: {POS_PATH} klasörü yok!")
        return [], []

    count = 0
    for filename in os.listdir(POS_PATH):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            path = os.path.join(POS_PATH, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, SIZE)
                fd = compute_hog_descriptor(img, cell_size=(8,8), block_size=(2,2), num_bins=9)
                if len(fd) > 0:
                    data.append(fd)
                    labels.append(1) # 1 = Nesne Var
                    count += 1
    print(f"-> {count} adet Pozitif resim yüklendi.")

    # 2. Negatif (Nesne Yok) Resimleri Yükle
    print(f"Negatif resimler okunuyor...")
    if not os.path.exists(NEG_PATH):
        print(f"HATA: {NEG_PATH} klasörü yok!")
        return [], []

    count = 0
    for filename in os.listdir(NEG_PATH):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            path = os.path.join(NEG_PATH, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, SIZE)
                fd = compute_hog_descriptor(img, cell_size=(8,8), block_size=(2,2), num_bins=9)
                if len(fd) > 0:
                    data.append(fd)
                    labels.append(0) # 0 = Nesne Yok
                    count += 1
    print(f"-> {count} adet Negatif resim yüklendi.")

    return np.array(data), np.array(labels)

if __name__ == "__main__":
    print("Veri seti hazırlanıyor...")
    X, y = load_data()
    
    if len(X) == 0:
        print("Resim bulunamadı. Lütfen klasörleri kontrol et.")
    else:
        # Veriyi böl
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = None)
        
        print(f"SVM Modeli Eğitiliyor... (Eğitim Verisi: {len(X_train)}, Test Verisi: {len(X_test)})")
        model = LinearSVC(random_state=42, max_iter=2000)
        model.fit(X_train, y_train)
        
        # Tahmin Yap
        y_pred = model.predict(X_test)
        
        # --- DETAYLI METRİKLER ---
        print("\n" + "="*40)
        print("          MODEL PERFORMANS RAPORU          ")
        print("="*40)
        
        # 1. Accuracy
        acc = accuracy_score(y_test, y_pred)
        print(f"Genel Doğruluk (Accuracy): %{acc*100:.2f}")
        
        # 2. Classification Report (Precision, Recall, F1)
        print("\n--- Sınıflandırma Raporu ---")
        print(classification_report(y_test, y_pred, target_names=["Negatif (Yok)", "Pozitif (Var)"]))
        
        # 3. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("\n--- Confusion Matrix (Sayısal) ---")
        print(cm)
        
        # 4. Confusion Matrix Görselleştirme ve Kaydetme
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=["Negatif", "Pozitif"], 
                        yticklabels=["Negatif", "Pozitif"])
            plt.title('Confusion Matrix (Model Başarısı)')
            plt.ylabel('Gerçek Durum')
            plt.xlabel('Modelin Tahmini')
            
            save_path = os.path.join(RESULTS_PATH, "confusion_matrix.png")
            plt.savefig(save_path)
            print(f"\n[BİLGİ] Confusion Matrix görseli kaydedildi: {save_path}")
            # plt.show() # Eğer masaüstünde görüyorsan açılabilir, sunucuda kapatıyoruz
        except Exception as e:
            print(f"\n[UYARI] Grafik çizilemedi (Seaborn/Matplotlib hatası): {e}")

        # Modeli Kaydet
        if not os.path.exists("models"):
            os.makedirs("models")
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
            
        print(f"\nModel başarıyla kaydedildi: {MODEL_PATH}")