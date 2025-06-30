import cv2


def contour_height(contour):
    _, _, height, _ = cv2.boundingRect(contour)
    return height


def brick_or_plate(piece, threshold=3, debug=False):
    top_key = -1
    top_coord = 0
    height = piece.base_image.shape[0]
    width = piece.base_image.shape[1]

    for key, value in piece.centroids.items():
        if value[1] > top_coord:
            top_coord = value[1]
            top_key = key

    h = contour_height(piece.contours[top_key])
    dist = abs(height - top_coord)

    ratio = abs(dist / h)
    ret = ratio < threshold

    if debug:
        print(f'Coord mes alta: {top_coord}')
        print(f'Alçada imatge: {height}')
        print(f'Pixels alçada més baix: {h}')
        print(f'Distància fins tope: {dist}')
        print(f'Ratio: {ratio} -> {ret}')
    '''
    if show_plot:
        # Plot the base image
        base_image = piece.base_image
        plt.imshow(cv2.cvtColor(base_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        for key, coord in piece.centroids.items():
            adjusted_x = coord[0]  # X-coordinates don't change
            adjusted_y = coord[1]  # Y-coordinates remain top-down for OpenCV
            plt.scatter(adjusted_x, adjusted_y, c='red', s=50)
            plt.text(adjusted_x, adjusted_y, f'{key}', color='blue', fontsize=10, ha='center', va='bottom')

        plt.title('Centroides')
        plt.show()'''

    return 'plate' if ret else 'brick'




