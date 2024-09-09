import numpy as np
import matplotlib.pyplot as plt
from utils import (
    crop_image_from_center,
    translate_image,
    numpyarr_to_img,
    adjust_average_pixel_value,
    display_image_opencv,
)
from ncc import ncc2
import skimage as sk
import skimage.io as skio
from image_similarity import custom_metric, custom_edge_metric, canny_edge_detection

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)


class Aligner:
    def dummy_align(self, base_img, target_img):
        """Handy for testing purposes. Returns the base image."""
        return base_img

    def simple_align(self, base_img, target_img):
        """Searches over a window of [-N,N] possible translational pixel displacements."""
        assert base_img.shape == target_img.shape

        ### User defined parameters
        N = 15
        SEARCH_GRID_CIRCUMRADIUS = 15

        # Compare a large as possible window of pixels
        SEARCH_GRID_CIRCUMRADIUS = min(
            SEARCH_GRID_CIRCUMRADIUS,
            base_img.shape[0] // 2 - N,
            base_img.shape[1] // 2 - N,
        )

        img_width, img_height = base_img.shape
        center_x = img_width // 2
        center_y = img_height // 2

        best_alignment_score = np.inf
        best_alignment = (0, 0)
        for i in range(-N, N + 1):
            for j in range(-N, N + 1):

                cropped_base_img = crop_image_from_center(
                    base_img,
                    center=(center_x + i, center_y + j),
                    circumradius=SEARCH_GRID_CIRCUMRADIUS,
                )

                cropped_target_img = crop_image_from_center(
                    target_img,
                    center=(center_x, center_y),
                    circumradius=SEARCH_GRID_CIRCUMRADIUS,
                )

                #cropped_base_img = adjust_average_pixel_value(
                #    cropped_base_img, cropped_target_img
                #)

                croppped_base_edges = canny_edge_detection(cropped_base_img)
                croppped_target_edges = canny_edge_detection(cropped_target_img)

                # fig, axs = plt.subplots(1, 5, figsize=(15, 5)) # Debugging
                # axs[0].imshow(cropped_base_img, cmap="gray")
                # axs[0].set_title("Cropped Base Image")

                # axs[1].imshow(cropped_target_img, cmap="gray")
                # axs[1].set_title("Cropped Target Image")

                # zero_img = np.zeros_like(cropped_base_img)
                # axs[2].imshow(zero_img, cmap="gray")
                # axs[2].set_title("Cropped Base Image with translation")

                # axs[3].imshow(croppped_base_edges, cmap="gray")
                # axs[3].set_title("Cropped Base Image modified")

                # axs[4].imshow(croppped_target_edges, cmap="gray")
                # axs[4].set_title("Cropped Target Image modified")
                # plt.tight_layout()
                # plt.show()

                #display_image_opencv(cropped_base_img, 6)
                #display_image_opencv(cropped_target_img, 6)
                #display_image_opencv(croppped_base_edges, 6)
                #display_image_opencv(croppped_target_edges, 6)

                # weighted sum on both "regular" and edge images
                similarity_score = 1.0 * custom_metric(
                    cropped_base_img, cropped_target_img
                ) + 0.0 * custom_edge_metric(croppped_base_edges, croppped_target_edges)

                if similarity_score < best_alignment_score:
                    best_alignment = (i, j)
                    best_alignment_score = similarity_score
                    print(f"***Dissimiliarity score: {best_alignment_score} at: {-j, i}")

        print(f"best alignment at coords: {-best_alignment[1], best_alignment[0]}")
        aligned_image = translate_image(base_img, -best_alignment[1], best_alignment[0])

        return aligned_image


def gaussian_normalize_to_01(img):
    # normalize according to gaussian
    mean_val = np.mean(img)
    std_val = np.std(img)
    zscore_img = (img - mean_val) / std_val

    # Rescale [0, 1]
    min_zscore = np.min(zscore_img)
    max_zscore = np.max(zscore_img)
    normalized_img = (zscore_img - min_zscore) / (max_zscore - min_zscore)

    return normalized_img
