# Frequencies
import cv2
import numpy as np
from matplotlib import pyplot as plt

def remove_sine_noise_peak_detection(image_path):
    img = cv2.imread(image_path, 0)

    if img is None:
        print("Could not open or find the image")
        return

    scale = 600 / img.shape[1]
    dim = (600, int(img.shape[0] * scale))
    img = cv2.resize(img, dim)

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    temp_mag = magnitude.copy()
    
    cv2.circle(temp_mag, (ccol, crow), 5, 0, -1)
    _, maxVal, _, maxLoc = cv2.minMaxLoc(temp_mag)
    
    noise_x, noise_y = maxLoc

    mask = np.ones((rows, cols, 2), np.uint8)
    
    r_remove = 5
    cv2.circle(mask, (noise_x, noise_y), r_remove, (0, 0), -1)
    
    sym_x = ccol + (ccol - noise_x)
    sym_y = crow + (crow - noise_y)
    cv2.circle(mask, (sym_x, sym_y), r_remove, (0, 0), -1)

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_clean = np.uint8(img_back)
    median = cv2.medianBlur(img_clean, 5)
    
    _, thresh = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plt.figure(figsize=(12, 8))
    
    mag_log = 20 * np.log(magnitude + 1)
    plt.subplot(2, 2, 1)
    plt.imshow(mag_log, cmap='gray')
    plt.title("1. Frequency Spectrum")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(mask[:,:,0] * 255, cmap='gray') 
    plt.title(f"2. Mask")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(img_clean, 'gray')
    plt.title("3. Image after filtering")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(opening, 'gray')
    plt.title(f"4. Result Count: {len(contours)} grains")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

remove_sine_noise_peak_detection('rice_sine.png')