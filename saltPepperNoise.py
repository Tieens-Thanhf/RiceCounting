# Median Blur
import cv2
import numpy as np
from matplotlib import pyplot as plt

def count_salt_pepper_simple(image_path):
    img = cv2.imread(image_path, 0)

    if img is None:
        print("Could not open or find the image")
        return

    scale = 600 / img.shape[1]
    dim = (600, int(img.shape[0] * scale))
    img = cv2.resize(img, dim)

    median = cv2.medianBlur(img, 5)

    val, thresh = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = len(contours)

    titles = ['Original Image', 'Median Blur', 'Otsu Threshold', 'Result']
    images = [img, median, thresh, opening]
    
    plt.figure(figsize=(15, 5))
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.suptitle(f"Result Count: {count} grains")
    plt.show()

count_salt_pepper_simple('rice_Salt_Pepper.png')