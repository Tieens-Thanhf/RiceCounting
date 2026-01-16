# Gamma Correction
import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_rice_dark_improved(image_path):
    img = cv2.imread(image_path, 0)

    if img is None:
        print("Could not open or find the image")
        return
    
    scale = 600 / img.shape[1]
    dim = (600, int(img.shape[0] * scale))
    img = cv2.resize(img, dim)

    gamma = 0.5
    img_float = img.astype(float) / 255.0
    img_gamma = np.power(img_float, gamma)
    img_gamma = (img_gamma * 255).astype(np.uint8)

    median = cv2.medianBlur(img_gamma, 5)

    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    background = cv2.morphologyEx(median, cv2.MORPH_OPEN, kernel_bg)

    diff = cv2.subtract(median, background)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(diff)

    val, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_small = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=2)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = len(contours)
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("1. Original Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img_gamma, cmap='gray')
    plt.title("2. After Gamma Correction")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(enhanced, cmap='gray')
    plt.title("3. After Background Subtraction & CLAHE")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(opening, cmap='gray')
    plt.title(f"4. Result Count: {count} grains")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

process_rice_dark_improved('rice_dark.png')