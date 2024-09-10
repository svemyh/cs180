# p1 - Images of the Russian Empire
# Sveinung Myhre <s.myhre@berkeley.edu>
#
# CS180/CS280 - starter code for project 1, available at: https://inst.eecs.berkeley.edu/~cs180/fa24/hw/proj1/

import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt

from utils import Aligner, remove_borders, translate_image, display_pyramid


INPUT_IMAGE = "./data/church.tif"
OUTPUT_IMAGE_PATH = "./output/out_colourized.jpg"


def generate_pyramid(img, smallest_size=32):
    """Generates image pyramid, halving image resolution each step, down to the smallest given size."""

    pyramid = [img]
    while True:
        img_width, img_height = img.shape
        img = sk.transform.resize(
            img, (img_width // 2, img_height // 2)
        )  # halve image res
        resized_img_width, resized_img_height = img.shape

        if min(resized_img_width, resized_img_height) < smallest_size:
            break
        pyramid.append(img)

    pyramid.reverse()
    return pyramid  # smallest image first


def align_using_pyramids(base_pyramid, target_pyramid):
    """Returns base_pyramid aligned to target_pyramid."""

    assert len(base_pyramid) == len(target_pyramid), "Pyramids must be of same shape"

    best_alignment = (0, 0)
    for i in range(len(base_pyramid)):
        best_alignment = (best_alignment[0] * 2, best_alignment[1] * 2)

        next_img, next_alignment = Aligner.simple_align(
            translate_image(base_pyramid[i], best_alignment[0], best_alignment[1]),
            target_pyramid[i],
            N=15,
            search_grid_circumradius=20,
        )

        best_alignment = (
            best_alignment[0] + next_alignment[0],
            best_alignment[1] + next_alignment[1],
        )

    print(f"Optimal alignment found: dx={best_alignment[0]}, dy={best_alignment[1]}")

    aligned_base_img = translate_image(
        base_pyramid[len(base_pyramid) - 1], best_alignment[0], best_alignment[1]
    )
    return aligned_base_img


if __name__ == "__main__":

    ### Read image-data and separate into three colour channels 'r', 'g' and 'b'.
    im = skio.imread(INPUT_IMAGE)
    height = int(np.floor(im.shape[0] / 3.0))
    im = sk.img_as_float(im)
    b = im[:height]
    g = im[height : 2 * height]
    r = im[2 * height : 3 * height]

    # display_pyramid(generate_pyramid(g))

    ar = align_using_pyramids(generate_pyramid(r), generate_pyramid(g))
    ab = align_using_pyramids(generate_pyramid(b), generate_pyramid(g))
    ag = g

    ar, ag, ab = remove_borders(ar, ag, ab)
    im_out = np.dstack([ar, ag, ab])  # Reconstruct the coloured image

    ### Save and display
    plt.imsave(OUTPUT_IMAGE_PATH, im_out)
    skio.imshow(im_out)
    skio.show()
