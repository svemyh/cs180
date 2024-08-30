# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from aligner import Aligner
from utils import display_images


INPUT_IMAGE = "data/cathedral.jpg"
OUTPUT_IMAGE_PATH = "output/cathedral_colourized.jpg"


im = skio.imread(INPUT_IMAGE)
# convert to double (might want to do this later on to save memory)
im = sk.img_as_float(im)
# compute height of each part as simply 1/3 of total height
height = np.floor(im.shape[0] / 3.0).astype(np.int32)

# separate the color channels
r = im[2 * height : 3 * height]
g = im[height : 2 * height]
b = im[:height]


aligner = Aligner()
ar = aligner.dummy_align(r, b) # aligning image 'r' to a position as similar as possible to 'b' 
ag = aligner.dummy_align(g, b)

# Creating color image by assembling the three colour channels red, green and blue.
im_out = np.dstack([ar, ag, b])

display_images(r, g, b, im_out)  # For debugging

plt.imshow(im_out)
plt.title("Colorized image")
plt.show()

plt.imsave(OUTPUT_IMAGE_PATH, im_out)
