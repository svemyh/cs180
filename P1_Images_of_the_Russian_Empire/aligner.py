import numpy as np

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)


class Aligner:
    def dummy_align(self, image, target_image):
        """For testing purposes"""
        return image

    def align(self, img, target_img):
        """Searches over a window of [-N,N] possible translational pixel displacements."""
        assert img.shape == target_img.shape
        N = 1
        img_width = img.shape[0]
        img_height = img.shape[1]

        best_alignment_score = np.inf
        best_alignment = (0, 0)
        for i in range(-N, N):
            for j in range(-N, N):
                print(f"i: {i}, j: {j}")

                cropped_img = crop_image(
                    img,
                    top_left=(N, N),
                    bottom_right=(img_height - N, img_width - N),
                )

                cropped_target_img = crop_image(
                    target_img,
                    top_left=(N, N),
                    bottom_right=(img_height - N, img_width - N),
                )

                print(f'cropped_img shape: {cropped_img.shape}')
                print(f'cropped_target_img shape: {cropped_target_img.shape}')

                similiarity_score = l2(cropped_img, cropped_target_img)
                if similiarity_score < best_alignment_score:
                    best_alignment = (i, j)
                    best_alignment_score = similiarity_score

        aligned_image = crop_image(
            img,
            top_left=(N + best_alignment[0], N + best_alignment[1]),
            bottom_right=(
                img_height - N + best_alignment[0],
                img_width - N + best_alignment[1],
            ),
        )

        return aligned_image


def l2(img1, img2):
    assert img1.shape == img2.shape

    difference = img1 - img2
    l2_norm = np.sqrt(np.sum(np.square(difference)))

    return l2_norm


def crop_image(img, top_left, bottom_right):
    """Crop image using top-left and bottom-right coordinates."""

    top_row, left_col = top_left
    bottom_row, right_col = bottom_right

    cropped_img = img[top_row:bottom_row, left_col:right_col]
    return cropped_img
