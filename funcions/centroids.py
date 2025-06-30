import numpy as np


def calculate_centroids(contours_dict):
    centroids_dict = {}
    for contour_id, contour in contours_dict.items():
        if len(contour) > 0:
            if contour.ndim == 3 and contour.shape[2] == 2:
                x_coords = contour[:, 0, 0]
                y_coords = contour[:, 0, 1]
            elif contour.ndim == 2 and contour.shape[1] == 2:
                x_coords = contour[:, 0]
                y_coords = contour[:, 1]
            else:
                raise ValueError(f"Contor no valid: {contour.shape}")

            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            centroids_dict[contour_id] = (centroid_x, centroid_y)
        else:
            centroids_dict[contour_id] = None
    return centroids_dict
