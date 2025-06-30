import numpy as np

def find_grid_patterns_aprox(clusters, tolerance=10):
    centroids = np.array(list(clusters.values()))
    if centroids.ndim == 1:
        centroids = np.expand_dims(centroids, axis=0)

    x_rounded = np.round(centroids[:, 0] / tolerance) * tolerance
    y_rounded = np.round(centroids[:, 1] / tolerance) * tolerance

    unique_x = np.unique(x_rounded)
    unique_y = np.unique(y_rounded)

    rows = len(unique_y)
    columns = len(unique_x)

    return [rows, columns]


def find_closest_pair(original_pair, list_of_pairs):
    def shape_distance(pair1, pair2):
        area1, area2 = pair1[0] * pair1[1], pair2[0] * pair2[1]
        aspect1, aspect2 = pair1[0] / pair1[1], pair2[0] / pair2[1]

        area_diff = abs(area1 - area2)
        aspect_diff = abs(aspect1 - aspect2)

        return area_diff + 0.5 * aspect_diff

    closest_pair = None
    smallest_distance = float('inf')

    for pair in list_of_pairs:
        distance = shape_distance(original_pair, pair)
        if distance < smallest_distance:
            smallest_distance = distance
            closest_pair = pair

    return closest_pair

