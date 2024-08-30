import matplotlib.pyplot as plt
import numpy as np


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


def translate_image(img, row_offset: int, col_offset: int):
    """Translate an image given offsets in x and y. Retains img-dimensions between input- and output-image by either adding black padding or dropping pixels."""
    print(f"Original image shape: {img.shape}")
    rows, cols = img.shape
    translated_img = np.zeros((rows, cols), dtype=img.dtype)  # Empty image of same size

    if row_offset >= 0:
        src_rows = slice(0, rows - row_offset)
        dst_rows = slice(row_offset, rows)
    else:
        src_rows = slice(-row_offset, rows)
        dst_rows = slice(0, rows + row_offset)

    if col_offset >= 0:
        src_cols = slice(0, cols - col_offset)
        dst_cols = slice(col_offset, cols)
    else:
        src_cols = slice(-col_offset, cols)
        dst_cols = slice(0, cols + col_offset)

    translated_img[dst_rows, dst_cols] = img[src_rows, src_cols]

    print(f"Translated image shape: {translated_img.shape}")

    return translated_img

