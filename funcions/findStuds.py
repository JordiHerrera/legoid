import cv2
import numpy as np
import matplotlib.pyplot as plt


def regions(binary_image, debug=False,
            noise_kernel_size=(3, 3), noise_iterations=2,
            dilation_kernel_size=(11, 11), dilation_iterations=1):

    noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, noise_kernel_size)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, noise_kernel, iterations=noise_iterations)

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilation_kernel_size)
    dilated_image = cv2.dilate(cleaned_image, dilation_kernel, iterations=dilation_iterations)

    num_features, labeled_array = cv2.connectedComponents(dilated_image)

    if debug:
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(binary_image, cmap="gray")
        plt.title("Binari otiginal")

        plt.subplot(1, 3, 2)
        plt.imshow(cleaned_image, cmap="gray")
        plt.title("Eliminació soroll")

        plt.subplot(1, 3, 3)
        plt.imshow(labeled_array, cmap="nipy_spectral")
        plt.title(f"Components: {num_features - 1} regions diferenciades")
        plt.colorbar()
        plt.show()

    regs = {}
    for region_num in range(1, num_features):
        regs[region_num] = np.argwhere(labeled_array == region_num)

    return regs

def filtrar_regions(regions_dict, threshold_percentage=6, max_percent=65, debug=False):

    areas={}

    for key, value in regions_dict.items():
        areas[key] = len(value)

    min_area = 0
    max_area = 0

    for key, value in areas.items():
        if debug:
            print(f"ID {key}: Area = {value}")
        if value > max_area:
            max_area = value

    filtered = {}

    for key, value in areas.items():
        normalized = 100 * (value - min_area) / (max_area - min_area)
        if debug:
            print(f"ID {key}: Norm_Area = {normalized}")
        if normalized > threshold_percentage and normalized < max_percent:
            filtered[key] = regions_dict[key]

    if debug:
        print(f'Valors escollits')
        for key, value in filtered.items():
            normalized = 100 * (len(value) - min_area) / (max_area - min_area)
            print(f'ID {key}: {normalized}')
    return filtered
