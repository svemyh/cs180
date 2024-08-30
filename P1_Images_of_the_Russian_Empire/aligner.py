import numpy as np

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)


class Aligner:
    def dummy_align(self, base_img, target_img):
        """For testing purposes"""
        return base_img

    def simple_align(self, base_img, target_img):
        """Searches over a window of [-N,N] possible translational pixel displacements."""
        assert base_img.shape == target_img.shape

        N = 5
        SEARCH_GRID_CIRCUMRADIUS = 50

        search_grid_size = (
            2 * SEARCH_GRID_CIRCUMRADIUS + 1
        )  # Height & width of the search grid chosen from center of image. Must be odd.

        img_width, img_height = base_img.shape
        center_x = img_width // 2
        center_y = img_height // 2

        best_alignment_score = np.inf
        best_alignment = (0, 0)
        for i in range(-N, N+1):
            for j in range(-N, N+1):
                print(f"i: {i}, j: {j}")

                cropped_base_img = crop_image_with_center(
                    base_img,
                    center=(center_x+i, center_y+j),
                    circumradius=SEARCH_GRID_CIRCUMRADIUS,
                )

                cropped_target_img = crop_image_with_center(
                    target_img,
                    center=(center_x, center_y),
                    circumradius=SEARCH_GRID_CIRCUMRADIUS,
                )

                #print(f"cropped_img shape: {cropped_base_img.shape}")
                #print(f"cropped_target_img shape: {cropped_target_img.shape}")

                similiarity_score = l2_norm(cropped_base_img, cropped_target_img)
                if similiarity_score < best_alignment_score:
                    best_alignment = (i, j)
                    best_alignment_score = similiarity_score

        aligned_image = crop_image_with_center(
            base_img, center=(center_x+i, center_y+j), circumradius=SEARCH_GRID_CIRCUMRADIUS
        )
        return aligned_image


def l2_norm(img1, img2):
    assert img1.shape == img2.shape

    difference = img1 - img2
    l2_norm = np.sqrt(np.sum(np.square(difference)))

    return l2_norm


def crop_image_with_corners(img, top_left, bottom_right):
    """Crop image using top-left and bottom-right coordinates."""

    top_row, left_col = top_left
    bottom_row, right_col = bottom_right

    cropped_img = img[top_row:bottom_row, left_col:right_col]
    return cropped_img


def crop_image_with_center(img, center, circumradius: int):
    """Crop image by declaring a centr pixel coordinate then grabbing the 'circumradius' amount of pixels above it, below, to its right and elft."""
    center_row, center_col = center

    cropped_img = img[
        center_row - circumradius : center_row + circumradius,
        center_col - circumradius : center_col + circumradius,
    ]
    return cropped_img
