# Project 2 - Fun with Filters and Frequencies
# Sveinung Myhre <s.myhre@berkeley.edu>
# CS180/CS280 at UC Berkeley - Related material available at: https://inst.eecs.berkeley.edu/~cs180/fa24/

import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt

from scipy.signal import convolve2d

from utils import KernelGenerator

if __name__ == "__main__":
    # Part 1.1: Finite Difference Operator
    INPUT_IMAGE = "./data/cameraman.png"
    OUTPUT_PATH = "./output/result.jpg"

    img = skio.imread(INPUT_IMAGE, as_gray=True)
    img = sk.img_as_float(img)
    img = np.array(img)

    # Define the finite difference operator
    D = np.array([1, -1])
    D = 1 / 9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    kernelGenerator = KernelGenerator(kernel_width=3, kernel_height=3)

    ker = kernelGenerator.get_box_kernel()

    img_d = convolve2d(img, ker, mode="same")

    # Display & save results
    plt.imsave(OUTPUT_PATH, img_d)
    skio.imshow(img_d)
    skio.show()
