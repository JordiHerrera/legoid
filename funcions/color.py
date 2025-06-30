import cv2
import numpy as np
import pandas as pd
import colorsys
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
import matplotlib.pyplot as plt


def double_check_gray(hex, black=70, white=160, debug=False):
    rgb = hex2rgb(hex)

    normalized_rgb = [value / 255.0 for value in rgb]
    hsv = colorsys.rgb_to_hsv(*normalized_rgb)

    new_hsv = (hsv[0], 0, hsv[2])

    new_rgb = colorsys.hsv_to_rgb(*new_hsv)

    denormalized_rgb = tuple(int(value * 255) for value in new_rgb)

    val = denormalized_rgb[0]

    if debug:
        print(f'Double check gris: {val} ({black}, {white})')

    if val > white:
        return 'White'
    elif val < black:
        return 'Black'
    else:
        return 'Gray'


def increase_lightness(rgb, factor=0.2):
    r, g, b = [x / 255.0 for x in rgb]

    h, l, s = colorsys.rgb_to_hls(r, g, b)

    l = min(1, l + factor)

    r, g, b = colorsys.hls_to_rgb(h, l, s)

    r, g, b = [int(x * 255) for x in (r, g, b)]

    return (r, g, b)


def get_mean_hex(image, debug=False):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean_color = np.mean(image_rgb, axis=(0, 1))
    mean_color_hex = "{:02x}{:02x}{:02x}".format(int(mean_color[0]), int(mean_color[1]), int(mean_color[2])).upper()

    if debug:
        print(f'Codi color HEX detectat: {mean_color_hex}')

    return mean_color_hex


def get_dominant_hex(image, k=1, quantize_colors=16, debug=False):
    if not isinstance(k, int) or k <= 0:
        raise ValueError("La K es negativa")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    blurred = cv2.GaussianBlur(image, (33, 33), 0)

    hsv_image = cv2.cvtColor(blurred, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)
    s = cv2.multiply(s, 1.5)
    s = cv2.min(s, 255)
    enhanced_hsv = cv2.merge((h, s, v))
    enhanced_rgb = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2RGB)

    lab_image = rgb2lab(enhanced_rgb)
    lab_pixels = lab_image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=quantize_colors, random_state=42)
    kmeans.fit(lab_pixels)
    quantized_lab = kmeans.cluster_centers_[kmeans.labels_].reshape(lab_image.shape)
    quantized_rgb = (np.clip(lab2rgb(quantized_lab) * 255, 0, 255)).astype(np.uint8)


    if debug:
        print('a dins!')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(blurred)
        axes[0].set_title("Difuminat")
        axes[0].axis('off')

        axes[1].imshow(enhanced_rgb)
        axes[1].set_title("Increment Satauració")
        axes[1].axis('off')

        axes[2].imshow(quantized_rgb)
        axes[2].set_title("Quantize")
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()

    # Reshape quantized pixels for final clustering
    quantized_pixels = quantized_rgb.reshape(-1, 3)

    # Perform KMeans clustering on quantized colors
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(quantized_pixels)

    # Extract dominant color
    dominant_color = kmeans.cluster_centers_[0].astype(int)

    # Convert to hex
    dominant_hex = "{:02x}{:02x}{:02x}".format(dominant_color[0], dominant_color[1], dominant_color[2])

    return dominant_hex

def hex2rgb(hex_code):
    if len(hex_code) == 6:
        ret_rgb = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        return ret_rgb
    return (-1, -1, -1)


def rgb2hex(r,g,b):
    return "{:02x}{:02x}{:02x}".format(r,g,b)


def closest_color_name(target_hex, df, debug=False):
    target_rgb = np.array(hex2rgb(target_hex)).reshape(1, 1, 3)
    target_lab = rgb2lab(target_rgb / 255.0)[0, 0]

    def rgb_to_lab(rgb):
        rgb_array = np.array(rgb).reshape(1, 1, 3)
        return rgb2lab(rgb_array / 255.0)[0, 0]

    df['lab'] = df['rgb'].apply(rgb_to_lab)

    df['distance'] = df['lab'].apply(lambda lab: np.linalg.norm(target_lab - lab))

    closest_row = df.loc[df['distance'].idxmin()]

    if debug:
        print(f"Target Lab: {target_lab} | Closest Match: {closest_row['name']} -> {closest_row['simplified']}")

    ret = closest_row['simplified']

    if ret in ['Black', 'Gray', 'White']:
        ret = double_check_gray(target_hex, debug=False)

    return ret


'''def get_uneven_color(img, redu=0.15):
    mean_color = cv2.mean(img)[:3]  # Ignore the alpha channel if present
    mean_b, mean_g, mean_r = mean_color

    if redu > 0.0:
        mean_b = mean_b * (1 - redu)
        mean_g = mean_g * (1 - redu)
        mean_r = mean_r * (1 - redu)

    # Clamp values to valid range [0, 255]
    mean_b = np.clip(mean_b, 0, 255)
    mean_g = np.clip(mean_g, 0, 255)
    mean_r = np.clip(mean_r, 0, 255)

    # Convert to hex format
    hex_color = "{:02x}{:02x}{:02x}".format(int(mean_r), int(mean_g), int(mean_b))

    return hex_color'''

def get_uneven_color(img, inc=0.3):
    mean_color = cv2.mean(img)[:3]
    mean_b, mean_g, mean_r = mean_color

    if inc > 0.0:
        mean_b = mean_b + (255 - mean_b) * inc
        mean_g = mean_g + (255 - mean_g) * inc
        mean_r = mean_r + (255 - mean_r) * inc

    mean_b = np.clip(mean_b, 0, 255)
    mean_g = np.clip(mean_g, 0, 255)
    mean_r = np.clip(mean_r, 0, 255)

    hex_color = "{:02x}{:02x}{:02x}".format(int(mean_r), int(mean_g), int(mean_b))

    return hex_color

