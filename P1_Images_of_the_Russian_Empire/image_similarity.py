import numpy as np
import matplotlib.pyplot as plt
from utils import crop_image_from_center, translate_image, numpyarr_to_img, adjust_average_pixel_value
from ncc import ncc2
import skimage as sk
import skimage.io as skio
import cv2

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from skimage.measure import pearson_corr_coeff, find_contours


def custom_metric(img1, img2):
    s = 0 # similarity-score as weighted sum of different metrics
    pcc, _ = pearson_corr_coeff(img1, img2)


    s += 0.5 * abs(mse(img1, img2))
    s += 0.5 * abs(nrmse(img1, img2))
    s += 1.0 * abs(ncc2(img1, img2))
    s += 1.0 * abs(pcc)


    ssim_score = ssim(img1, img2, data_range=max(img1.max(), img2.max()) - min(img1.min(), img2.min()))
    normalized_dissimilarity_ssim_score = (1 - ssim_score) / 2 # 0.0 means identical, 1.0 means completely different as opposed to ssim_score which is the opposite (1.0 means identical, -1 means completely different)
    s += 4.5 * normalized_dissimilarity_ssim_score



    return s

