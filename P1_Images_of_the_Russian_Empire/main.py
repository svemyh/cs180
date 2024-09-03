# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from aligner import Aligner
from utils import (
    get_3_colourchannels_boilerplate,
    display_images,
    display_images_noQt,
    display_image,
    display_image_opencv,
)
from skimage.transform import rescale, resize
from utils import translate_image, remove_borders
from PIL import Image


INPUT_IMAGE = "data/tobolsk.jpg"
OUTPUT_IMAGE_PATH = "output/out_colourized.jpg"


if __name__ == "__main__":
    r, g, b = get_3_colourchannels_boilerplate(INPUT_IMAGE)
    r, g, b = remove_borders(r, g, b)

    # aligning images 'r' and 'g' to a position as similar as possible to 'b'
    aligner = Aligner()
    ar = aligner.simple_align(r, b)
    ag = aligner.simple_align(g, b)

    # Creating color image by assembling the three colour channels red, green and blue.
    im_out_baseline = np.dstack([r, g, b])
    im_out = np.dstack([ar, ag, b])

    r_man = translate_image(r, 3, -2)
    g_man = translate_image(g, 2, 4)

    # r_man = translate_image(r, 2, -3)
    # g_man = translate_image(g, 2, 3)

    im_man = np.dstack([r_man, g_man, b])

    display_image_opencv(im_out_baseline, 3)
    display_image_opencv(im_out, 3)
    display_image_opencv(im_man, 3)

    plt.imsave(OUTPUT_IMAGE_PATH, im_out)
