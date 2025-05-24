import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

def get_lightness_channel(image):
    hsv = rgb2hsv(image)
    L = hsv[..., 2]
    return (L * 255).astype(np.uint8)

def log_contrast(img):
    img_norm = img / 255.0
    log_corrected = np.log1p(img_norm) / np.log1p(1.0)
    return (log_corrected * 255).astype(np.uint8)

def compute_ngtdm_features(img_gray, d=2, levels=8, return_map=False):
    quantized = np.floor(img_gray / (256 / levels)).astype(np.uint8)
    h, w = img_gray.shape
    P_i = np.zeros(levels, dtype=np.int32)
    S_i = np.zeros(levels, dtype=np.float64)
    total_pixels = 0
    ngtdm_map = np.zeros_like(img_gray, dtype=np.float64)

    for y in range(d, h - d):
        for x in range(d, w - d):
            center_val = quantized[y, x]
            neighborhood = quantized[y-d:y+d+1, x-d:x+d+1].astype(np.float64)
            neighborhood[d, d] = np.nan
            mean_val = np.nanmean(neighborhood)
            diff = abs(center_val - mean_val)
            if not np.isnan(diff):
                S_i[center_val] += diff
                P_i[center_val] += 1
                total_pixels += 1
                ngtdm_map[y, x] = diff

    p_i = P_i / total_pixels
    S_i[S_i == 0] = 1e-10

    coarseness = 1.0 / (np.sum(p_i * S_i) + 1e-10)
    contrast = np.sum(p_i[:, None] * p_i[None, :] * (np.arange(levels)[:, None] - np.arange(levels)[None, :])**2)
    busyness = np.sum(np.abs(np.arange(levels) - np.sum(p_i * np.arange(levels))) * p_i) / (np.sum(S_i) + 1e-10)

    if return_map:
        ngtdm_map = (ngtdm_map / np.max(ngtdm_map) * 255).astype(np.uint8)
        return coarseness, contrast, busyness, S_i, ngtdm_map
    else:
        return coarseness, contrast, busyness, S_i

def process_image(path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"result_{timestamp}"

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    L = get_lightness_channel(img)
    L_corrected = log_contrast(L)

    coarseness1, contrast1, busyness1, S_i1, ngtdm_map1 = compute_ngtdm_features(L, return_map=True)
    coarseness2, contrast2, busyness2, S_i2, ngtdm_map2 = compute_ngtdm_features(L_corrected, return_map=True)

    # Создание общего коллажа
    fig, axs = plt.subplots(4, 2, figsize=(12, 20))

    # 1. Полутоновое изображение
    axs[0, 0].imshow(L, cmap='gray')
    axs[0, 0].set_title("Полутоновое изображение")
    axs[0, 0].axis('off')

    # 2. Контрастированное полутоновое
    axs[0, 1].imshow(L_corrected, cmap='gray')
    axs[0, 1].set_title("Контрастированное полутоновое")
    axs[0, 1].axis('off')

    # 3. Гистограмма полутонового
    axs[1, 0].hist(L.ravel(), bins=256, color='gray')
    axs[1, 0].set_title("Гистограмма - Полутоновое")

    # 4. Гистограмма контрастированного
    axs[1, 1].hist(L_corrected.ravel(), bins=256, color='gray')
    axs[1, 1].set_title("Гистограмма - Контрастированное")

    # 5. График NGTDM до
    axs[2, 0].bar(np.arange(len(S_i1)), S_i1, color='gray')
    axs[2, 0].set_title("NGTDM S_i - Полутоновое")
    axs[2, 0].set_xlabel('Quantized Levels')
    axs[2, 0].set_ylabel('Sum of Differences')

    # 6. График NGTDM после
    axs[2, 1].bar(np.arange(len(S_i2)), S_i2, color='gray')
    axs[2, 1].set_title("NGTDM S_i - Контрастированное")
    axs[2, 1].set_xlabel('Quantized Levels')
    axs[2, 1].set_ylabel('Sum of Differences')

    # 7. Карта NGTDM до
    axs[3, 0].imshow(ngtdm_map1, cmap='gray')
    axs[3, 0].set_title("Карта разностей NGTDM - Полутоновое")
    axs[3, 0].axis('off')

    # 8. Карта NGTDM после
    axs[3, 1].imshow(ngtdm_map2, cmap='gray')
    axs[3, 1].set_title("Карта разностей NGTDM - Контрастированное")
    axs[3, 1].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.close()

    # Печать признаков
    print("\n=== Текстурные признаки ===")
    print(f"До преобразования:    Coarseness={coarseness1:.4f}, Contrast={contrast1:.4f}, Busyness={busyness1:.4f}")
    print(f"После преобразования: Coarseness={coarseness2:.4f}, Contrast={contrast2:.4f}, Busyness={busyness2:.4f}")

# Пример запуска
process_image("texture.png")
