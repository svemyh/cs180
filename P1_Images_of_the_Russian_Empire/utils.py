import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def display_images(r, g, b, im_out):
    """Displays a single large figure with 4 subplots."""

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs[0, 0].imshow(r, cmap="gray")
    axs[0, 0].set_title("r image")

    axs[0, 1].imshow(g, cmap="gray")
    axs[0, 1].set_title("g image")

    axs[1, 0].imshow(b, cmap="gray")
    axs[1, 0].set_title("b image")

    axs[1, 1].imshow(im_out)
    axs[1, 1].set_title("Colorized image")

    # Add color bars for the grayscale images
    cbar_r = plt.colorbar(
        axs[0, 0].imshow(r, cmap="gray"), ax=axs[0, 0], orientation="vertical"
    )
    cbar_g = plt.colorbar(
        axs[0, 1].imshow(g, cmap="gray"), ax=axs[0, 1], orientation="vertical"
    )
    cbar_b = plt.colorbar(
        axs[1, 0].imshow(b, cmap="gray"), ax=axs[1, 0], orientation="vertical"
    )

    plt.tight_layout()

    plt.show()


# # Debug the indivudidual images
# plt.imshow(r, cmap="gray")
# plt.title("r image")
# plt.colorbar()
# plt.show()
# plt.imshow(g, cmap="gray")
# plt.title("g image")
# plt.colorbar()
# plt.show()
# plt.imshow(b, cmap="gray")
# plt.title("b image")
# plt.colorbar()
# plt.show()


# # Save and display image
# plt.imshow(im_out)
# plt.title("Colorized image")
# plt.show()


def crop_image_with_corners(img, top_left, bottom_right):
    """Crop image using top-left and bottom-right coordinates."""

    top_row, left_col = top_left
    bottom_row, right_col = bottom_right

    cropped_img = img[top_row:bottom_row, left_col:right_col]
    return cropped_img


def crop_image_from_center(img, center, circumradius: int):
    """Crop image by declaring a centr pixel coordinate then grabbing the 'circumradius' amount of pixels above it, below, to its right and elft."""
    center_row, center_col = center

    cropped_img = img[
        center_row - circumradius : center_row + circumradius,
        center_col - circumradius : center_col + circumradius,
    ]
    return cropped_img


def translate_image(img, x_offset: int, y_offset: int):
    """Translate an image given x and y offsets.
    Retains img-dimensions between input- and output-image by either adding black padding or dropping pixels.
    Positive x_offset means translating to the right (dropping pixels on the rightmost side and adding black padding on the left side), positive y_offset means translating upwards.
    """

    rows, cols = img.shape
    translated_img = np.zeros((rows, cols), dtype=img.dtype)  # Empty image of same size

    x_offset = x_offset * (-1)

    if y_offset >= 0:
        src_rows = slice(y_offset, rows)
        dst_rows = slice(0, rows - y_offset)
    else:
        src_rows = slice(0, rows + y_offset)
        dst_rows = slice(-y_offset, rows)

    if x_offset >= 0:
        src_cols = slice(x_offset, cols)
        dst_cols = slice(0, cols - x_offset)
    else:
        src_cols = slice(0, cols + x_offset)
        dst_cols = slice(-x_offset, cols)

    translated_img[dst_rows, dst_cols] = img[src_rows, src_cols]

    return translated_img


def numpyarr_to_img(image_array):
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    image = Image.fromarray(image_array, mode='L')
    return image


def adjust_average_pixel_value(source_img, target_img):
    avg_source = np.mean(source_img)
    avg_target = np.mean(target_img)
    
    difference = avg_target - avg_source
    adjusted_img = source_img + difference
    adjusted_img = np.clip(adjusted_img, 0, 1)
    
    return adjusted_img