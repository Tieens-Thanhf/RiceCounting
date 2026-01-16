import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def count_rice_basic(image_path):
    img = cv.imread(image_path, 0)

    if img is None:
        print("Could not open or find the image")
        return
    
    scale = 600 / img.shape[1]
    dim = (600, int(img.shape[0] * scale))
    img_resized = cv.resize(img, dim)

    blur = cv.GaussianBlur(img_resized, (5, 5), 0)

    val, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

    contours, hierarchy = cv.findContours(opening, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    count = len(contours)

    result_img = cv.cvtColor(img_resized, cv.COLOR_GRAY2BGR)
    cv.drawContours(result_img, contours, -1, (0, 255, 0), 2)

    titles = ['Original Image', 'Blurred Image', 'Thresholded Image', 'Morphological Opening', 'Detected Grains']
    images = [img_resized, blur, thresh, opening, result_img]

    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        if i == 4:
            plt.imshow(cv.cvtColor(images[i], cv.COLOR_BGR2RGB))
        else:
            plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.axis('off')

    plt.suptitle(f"Total Rice Grains Counted: {count}")
    plt.show()

count_rice_basic('rice.png')