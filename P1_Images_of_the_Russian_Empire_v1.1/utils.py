import numpy as np


class Aligner:
    def dummy_align(self, base_img, target_img):
        """Handy for testing purposes. Returns the base image."""
        return base_img

    def simple_align(_img_1, _img_2, N=15):
        """Searches over a window of [-N,N] possible translational pixel displacements."""
        best_value = float("inf")
        best_dx, best_dy = 0, 0

        img_1 = crop_borders(_img_1, 0.1)
        img_2 = crop_borders(_img_2, 0.1)

        for dx in range(-N, N):
            for dy in range(-N, N):
                ssd = custom_metric(img_translate(img_1, dx, dy), img_2)
                if ssd < best_value:
                    best_value = ssd
                    best_dx, best_dy = dx, dy

        print("Alignment:")
        print(f"dx = {best_dx}")
        print(f"dy = {best_dy}")

        return img_translate(_img_1, best_dx, best_dy)


def crop_borders(img, percentage):
    x_crop = int(percentage * img.shape[0] / 2)
    y_crop = int(percentage * img.shape[1] / 2)
    return img[x_crop : img.shape[0] - x_crop, y_crop : img.shape[1] - y_crop]


def img_translate(img, dx, dy):
    result = np.roll(img, dx, axis=1)
    result = np.roll(result, dy, axis=0)
    return result


def custom_metric(img_1, img_2):
    score = 0
    score += sum_of_squared_diff(img_1, img_2)
    return score


def sum_of_squared_diff(img_1, img_2):
    return np.sum((img_1 - img_2) ** 2)


def remove_borders(r, g, b, percentage=0.9):
    """Remove borders such that only the overlapping parts remain. Solution could be more or less intelligent"""
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
    """Crop image by declaring a centr pixel coordinate then grabbing the 'circumradius' amount of pixels above it, below, to its right and elft."""
    center_row, center_col = center

    cropped_img = img[
        center_row - circumradius : center_row + circumradius,
        center_col - circumradius : center_col + circumradius,
    ]
    return cropped_img
