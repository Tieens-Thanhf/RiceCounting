import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_all_filters(image_path):
    img = cv2.imread(image_path, 0)

    if img is None:
        print("Could not open or find the image")
        return
    
    scale = 500 / img.shape[1]
    dim = (500, int(img.shape[0] * scale))
    img_resized = cv2.resize(img, dim)
    
    img_median = cv2.medianBlur(img_resized, 5)

    mean_brightness = np.mean(img_median)
    if mean_brightness < 100:
        gamma = 0.5
    else:
        gamma = 1.0
    img_float = img_median.astype(float) / 255.0
    img_gamma = np.power(img_float, gamma)
    img_gamma = (img_gamma * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(img_gamma)

    dft = cv2.dft(np.float32(img_clahe), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    
    rows, cols = img_clahe.shape
    crow, ccol = rows // 2, cols // 2
    
    temp_mag = magnitude.copy()
    cv2.circle(temp_mag, (ccol, crow), 5, 0, -1)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(temp_mag)

    mask = np.ones((rows, cols, 2), np.uint8)
    r_remove = 5
    cv2.circle(mask, maxLoc, r_remove, (0, 0), -1)
    sym_x = ccol + (ccol - maxLoc[0])
    sym_y = crow + (crow - maxLoc[1])
    cv2.circle(mask, (sym_x, sym_y), r_remove, (0, 0), -1)
    
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_fft = np.uint8(img_back)

    img_median_post = cv2.medianBlur(img_fft, 5)

    val, thresh = cv2.threshold(img_median_post, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = len(contours)

    titles = ['Original', 'Median Blur', 'Gamma Correction', 'CLAHE', 'FFT Denoise', 'Median Blur Post-FFT', 'Threshold', 'Morphology', f'Final Count: {count}']
    images = [img_resized, img_median, img_gamma, img_clahe, img_fft, img_median_post, thresh, opening, opening]

    plt.figure(figsize=(18, 12))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.suptitle(f"Combined Pipeline Result: {count} grains")
    plt.tight_layout()
    plt.show()
    
    return count

def process_multiple_images(image_list):
    for img in image_list:
        print(f"Processing {img}")
        process_all_filters(img)

if __name__ == "__main__":
    image_list = ['rice.png', 'rice_dark.png', 'rice_Salt_Pepper.png', 'rice_sine.png']
    process_multiple_images(image_list)