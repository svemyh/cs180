import matplotlib.pyplot as plt
import skimage.io as skio


def display_pyramid(pyramid):
    """Displays all images in the pyramid sequentially."""
    for i in pyramid:
        skio.imshow(i)
        skio.show()
