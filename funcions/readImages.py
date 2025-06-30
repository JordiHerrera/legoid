from classes.legoPiece import LegoPiece

import cv2
import matplotlib.pyplot as plt
import numpy as np

import tkinter as tk
from tkinter import filedialog


def select_image_file(string_dialeg):
    root = tk.Tk()
    root.withdraw()

    file_types = [("Imatges", "*.png *.jpg *.jpeg")]

    file_path = filedialog.askopenfilename(title=string_dialeg, filetypes=file_types)

    if file_path:
        print(f"Fitxer seleccionat: {file_path}")
        return file_path
    else:
        print("Cap fitxer seleccionat")
        return ""

def remove_bg(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)

    try:
        cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv2.contourArea)[-1]

        x, y, w, h = cv2.boundingRect(cnt)
        dst = img[y:y + h, x:x + w]

        return dst
    except:
        return img

def get_subimages(image, debug=False):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    v = cv2.equalizeHist(v)
    enhanced_hsv = cv2.merge((h, s, v))
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)

    adaptive_thresholded = cv2.adaptiveThreshold(
        gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    if debug:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Contrast")
        plt.imshow(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Thresholded")
        plt.imshow(adaptive_thresholded, cmap="gray")
        plt.axis("off")
        plt.show()

    contours, _ = cv2.findContours(adaptive_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 500   # area minima 500 (passar a umbral per diferents resolucions?)
    contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]

    lego_pieces = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        subimage = image[y:y + h, x:x + w]
        thresholded_subimage = adaptive_thresholded[y:y + h, x:x + w]

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour - [x, y]], -1, 255, thickness=cv2.FILLED)

        lego_piece = LegoPiece(subimage, thresholded_subimage)
        lego_piece.nobg_image = remove_bg(subimage)
        lego_pieces.append(lego_piece)

    return lego_pieces


def threshold_v2(image, debug=False):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)

    adaptive_thresholded = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                 cv2.THRESH_BINARY_INV, 11, 2)

    if debug:
        if debug:
            plt.figure(figsize=(10, 5))
            plt.title("Thresholded Image")
            plt.imshow(gray_image, cmap="gray")
            plt.axis("off")
            plt.show()

    return adaptive_thresholded


def correccio_calid(image, debug=False):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    l, a, b = cv2.split(lab_image)

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    a = np.clip(a - 0.8 * (a_mean - 128), 0, 255).astype(np.uint8)
    b = np.clip(b - 0.8 * (b_mean - 128), 0, 255).astype(np.uint8)

    corrected_lab = cv2.merge((l, a, b))

    corrected_image = cv2.cvtColor(corrected_lab, cv2.COLOR_Lab2BGR)

    if debug:
        image1_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(image1_rgb)
        plt.title('Image 1')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(image2_rgb)
        plt.title('Image 2')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return corrected_image