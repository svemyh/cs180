import numpy as np
from utils import crop_image_from_center, translate_image
from ncc import ncc
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

        ### User defined parameters
        N = 15
        SEARCH_GRID_CIRCUMRADIUS = 100

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

                similiarity_score = ncc(cropped_base_img, cropped_target_img)
                if similiarity_score < best_alignment_score:
                    best_alignment = (i, j)
                    best_alignment_score = similiarity_score

                    

        print(f"best alignment at coords: {best_alignment}"
                    )
        aligned_image = translate_image(base_img, best_alignment[0], -best_alignment[1])

        return aligned_image


def l2_norm(img1, img2):
    assert img1.shape == img2.shape

    difference = img1 - img2
    l2_norm = np.sqrt(np.sum(np.square(difference)))

    return l2_norm



