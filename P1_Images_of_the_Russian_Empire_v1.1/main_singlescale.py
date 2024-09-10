# p1 - Images of the Russian Empire
# Sveinung Myhre <s.myhre@berkeley.edu>
#
# CS180/CS280 - starter code for project 1, available at: https://inst.eecs.berkeley.edu/~cs180/fa24/hw/proj1/

import numpy as np
import skimage as sk
import skimage.io as skio
from utils import Aligner, remove_borders
import matplotlib.pyplot as plt


INPUT_IMAGE = "./data/cathedral.jpg"
OUTPUT_IMAGE_PATH = "./output/out_colourized.jpg"

if __name__ == "__main__":

    ### Read image-data and separate into three colour channels 'r', 'g' and 'b'.
    im = skio.imread(INPUT_IMAGE)
    height = int(np.floor(im.shape[0] / 3.0))
    im = sk.img_as_float(im)
    b = im[:height]
    g = im[height : 2 * height]
    r = im[2 * height : 3 * height]

    ### Align
    ar = Aligner.simple_align(r, g, N=15, search_grid_circumradius=20)
    ag = g
    ab = Aligner.simple_align(b, g, N=15, search_grid_circumradius=20)

    ar, ag, ab = remove_borders(ar, ag, ab)
    im_out = np.dstack([ar, ag, ab])  # Reconstruct the coloured image

    ### Save and display
    plt.imsave(OUTPUT_IMAGE_PATH, im_out)
    skio.imshow(im_out)
    skio.show()
