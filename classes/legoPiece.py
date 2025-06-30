import cv2
import matplotlib.pyplot as plt


class LegoPiece:
    def __init__(self, base_image, thresholded_image):
        self.base_image = base_image
        self.thresholded_image = thresholded_image
        self.centroids = []
        self.color_simple = ''
        self.color_hex = ''
        self.contours = {}
        self.grid_pattern = [-1, -1]
        self.stud_count = 0
        self.is_plate = 'brick'
        self.nobg_image = []
        self.given_name = ''
        self.is_uneven = False

    def show_images(self):
        plt.figure(figsize=(10, 5))

        # Display base image
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(self.base_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Base Image")

        # Display thresholded image
        plt.subplot(1, 2, 2)
        plt.imshow(self.thresholded_image, cmap="gray")
        plt.axis("off")
        plt.title("Thresholded Image")

        plt.tight_layout()
        plt.show()

def print_piece(piece):
    print(f'Grid pattern: {piece.grid_pattern[0]} x {piece.grid_pattern[1]}')
    print(f'Color: {piece.color_simple}')
    print(f'Tipus: {piece.is_plate}')
    print(f'Type: {type}')
