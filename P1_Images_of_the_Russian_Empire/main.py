# CS194-26 (CS294-26): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from aligner import Aligner
from utils import display_images
from skimage.transform import rescale, resize
from utils import translate_image


INPUT_IMAGE = "data/monastery.jpg"
OUTPUT_IMAGE_PATH = "output/out_colourized.jpg"


im = plt.imread(INPUT_IMAGE)
# convert to double (might want to do this later on to save memory)
im = sk.img_as_float(im)

# im = resize(im, (int(im.shape[0] / 2), int(im.shape[1] / 2)), anti_aliasing=True)

# compute height of each part as simply 1/3 of total height
height = np.floor(im.shape[0] / 3.0).astype(np.int32)

# separate the color channels
r = im[2 * height : 3 * height]
g = im[height : 2 * height]
b = im[:height]
zero = np.zeros_like(r)

aligner = Aligner()
# aligning images 'r' and 'g' to a position as similar as possible to 'b'
ar = aligner.simple_align(r, b)
ag = aligner.simple_align(g, b)

# Creating color image by assembling the three colour channels red, green and blue.
im_out_baseline = np.dstack([r, g, b])
im_out = np.dstack([ar, ag, b])

display_images(r, g, b, im_out_baseline) # For debugging
display_images(ar, ag, b, im_out)  

r_man = translate_image(r, 3, -2)
g_man = translate_image(g, 2, 4)
im_man = np.dstack([r_man, g_man, b])
display_images(r_man, g_man, b, im_man)


display_images(r, ar, zero, zero) # For debugging
display_images(g, ag, zero, zero)  
display_images(b, b, zero, zero)  


plt.figure(figsize=(8, 8))
plt.imshow(im_out_baseline)
plt.title("Colorized image (baseline)")

plt.figure(figsize=(8, 8))
plt.imshow(im_out)
plt.title("Colorized image")

plt.show()

plt.imsave(OUTPUT_IMAGE_PATH, im_out)
