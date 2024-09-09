# p1 - Images of the Russian Empire
# Sveinung Myhre <s.myhre@berkeley.edu>
#
# CS180/CS280 - starter code for project 1, available at: https://inst.eecs.berkeley.edu/~cs180/fa24/hw/proj1/

import numpy as np
import skimage as sk
import skimage.io as skio
from skimage.transform import pyramid_gaussian
from utils import Aligner, crop_borders, remove_borders

INPUT_IMAGE = "./data/cathedral.jpg"
OUTPUT_IMAGE_PATH = "./output/out_colourized.jpg"

if __name__ == "__main__":
    ### Boilerplate / setup
    im = skio.imread(INPUT_IMAGE)
    height = int(np.floor(im.shape[0] / 3.0))
    im = sk.img_as_float(im)
    b = im[:height]
    g = im[height : 2 * height]
    r = im[2 * height : 3 * height]

    ### Align
    ar = Aligner.simple_align(r, g)
    # ag = aligner.simple_align(g, g)
    ag = g
    ab = Aligner.simple_align(b, g)

    ar, ag, ab = remove_borders(ar, ag, ab)
    im_out = np.dstack([ar, ag, ab])

    ### Save and display
    skio.imshow(im_out)
    skio.show()
