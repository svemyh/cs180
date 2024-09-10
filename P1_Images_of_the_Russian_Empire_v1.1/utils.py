import numpy as np


class Aligner:
    def dummy_align(base_img, target_img):
        """Handy for testing purposes. Returns the base image."""
        return base_img

    def simple_align(base_img, target_img, N=15, search_grid_circumradius=np.inf):
        """Searches over a window of [-N,N] possible translational pixel displacements.


        Compare a large as possible window of pixels without encountering errors at
        the borders, by leaving search_grid_circumradius=np.inf.


        Alternatively choose a smaller search area selected from the center of the
        image. Width of the smaller search-square is given by 2 * search_grid_circumradius.
        """

        assert (
            base_img.shape == target_img.shape
        ), "Images are assumed to be of same shape."
        img_width, img_height = base_img.shape

        search_grid_circumradius = min(
            search_grid_circumradius,
            img_width // 2 - N,
            img_height // 2 - N,
        )

        base_img_cropped = crop_image_from_center(
            base_img, (img_width // 2, img_height // 2), search_grid_circumradius
        )
        target_img_cropped = crop_image_from_center(
            target_img, (img_width // 2, img_height // 2), search_grid_circumradius
        )

        best_alignment = (0, 0)
        # Init at max. Keeps track of current lowest image similarity score.
        best_alignment_score = np.inf
        for i in range(-N, N):
            for j in range(-N, N):
                dissimilarity_score = Metrics.l2_norm(
                    translate_image(base_img_cropped, x=i, y=j), target_img_cropped
                )
                if dissimilarity_score < best_alignment_score:
                    best_alignment_score = dissimilarity_score
                    best_alignment = (i, j)

        print(f"Alignment: {best_alignment}")

        return translate_image(base_img, best_alignment[0], best_alignment[1])


class Metrics:

    def l1_norm(img_1, img_2):
        return np.sum(np.abs(img_1 - img_2))

    def l2_norm(img_1, img_2):
        return np.sum((img_1 - img_2) ** 2)

    def linf_norm(img_1, img_2):
        return np.max(np.abs(img_1 - img_2))


def translate_image(img, x, y):
    """Positive x rolls image to the right, positve y rolls image downwards."""
    result = np.roll(img, x, axis=1)
    result = np.roll(result, y, axis=0)
    return result


def remove_borders(r, g, b, percentage=0.9):
    """Remove borders, keeping only a given perventage of the image, crpped from the center."""
    assert r.shape == g.shape == b.shape
    max_height, max_width = r.shape
    r = crop_image_from_center(
        r,
        (max_height // 2, max_width // 2),
        int(percentage * min(max_height, max_width) / 2),
    )
    g = crop_image_from_center(
        g,
        (max_height // 2, max_width // 2),
        int(percentage * min(max_height, max_width) / 2),
    )
    b = crop_image_from_center(
        b,
        (max_height // 2, max_width // 2),
        int(percentage * min(max_height, max_width) / 2),
    )
    return r, g, b


def crop_image_from_center(img, center, circumradius: int):
    """Crop image by declaring a centr pixel coordinate then grabbing
    the 'circumradius' amount of pixels above it, below, to its right and its elft."""

    center_row, center_col = center
    cropped_img = img[
        center_row - circumradius : center_row + circumradius,
        center_col - circumradius : center_col + circumradius,
    ]
    return cropped_img
