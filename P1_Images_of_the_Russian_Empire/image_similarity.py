import numpy as np
import matplotlib.pyplot as plt
from utils import (
    crop_image_from_center,
    translate_image,
    numpyarr_to_img,
    adjust_average_pixel_value,
)
from ncc import ncc2
import skimage as sk
import skimage.io as skio
import cv2

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from skimage.measure import pearson_corr_coeff, find_contours


def custom_metric(img1, img2):
    s = 0  # similarity-score as weighted sum of different metrics
    pcc, _ = pearson_corr_coeff(img1, img2)

    # s += 0.0 * abs(mse(img1, img2))
    #s += 1.0 * abs(nrmse(img1, img2))
    # s += 1.0 * abs(ncc2(img1, img2))
    #s += 0.0 * abs(pcc)
    s += 1.0 * ssim_metric(img1, img2)
    # s += 0.0 * sobel_metric(img1, img2, 1, 0)
    # s += 0.0 * sobel_metric(img1, img2, 0, 1)
    #s += 1 * mae(img1, img2)
    #s += 1.0 * l2_norm_mean(img1, img2)
    #s += 1.0 * ncc_v3(img1, img2)

    return s


def custom_edge_metric(img1, img2):
    img1 = img1 * 100
    img2 = img2 * 100

    s = 0  # similarity-score as weighted sum of different metrics
    pcc, _ = pearson_corr_coeff(img1, img2)

    # s += 0.0 * abs(mse(img1, img2))
    #s += 1.0 * abs(nrmse(img1, img2))
    # s += 1.0 * abs(ncc2(img1, img2))
    #s += 0.0 * abs(pcc)
    #s += 1.0 * ssim_metric(img1, img2)
    # s += 0.0 * sobel_metric(img1, img2, 1, 0)
    # s += 0.0 * sobel_metric(img1, img2, 0, 1)
    #s += 0.5 * mae(img1, img2)
    s += 1.0 * l2_norm_mean(img1, img2)
    # s += 1.0 * ncc_v3(img1, img2)

    return s


def ssim_metric(img1, img2):
    ssim_score = ssim(
        img1, img2, data_range=max(img1.max(), img2.max()) - min(img1.min(), img2.min())
    )
    normalized_dissimilarity_ssim_score = (
        1 - ssim_score
    ) / 2  # 0.0 means identical, 1.0 means completely different as opposed to ssim_score which is the opposite (1.0 means identical, -1 means completely different)
    return normalized_dissimilarity_ssim_score


def mae(img1, img2):
    score = np.mean(np.abs(img1 - img2))
    return score


def l2_norm_mean(img1, img2):
    assert img1.shape == img2.shape

    difference = img1 - img2
    l2_norm = np.mean(np.sqrt(np.sum(np.square(difference))))

    return l2_norm


def ncc_v0(img1, img2):
    norm_img1 = np.linalg.norm(img1)
    norm_img2 = np.linalg.norm(img2)

    normalized_image1 = img1 / norm_img1
    normalized_image2 = img2 / norm_img2

    ncc_score = np.sum(normalized_image1 * normalized_image2)
    return ncc_score

def ncc_v3(img1, img2):
    norm_img1 = np.linalg.norm(img1)
    norm_img2 = np.linalg.norm(img2)

    img1 = img1 - np.mean(img1)
    img2 = img2 - np.mean(img2)


    normalized_image1 = img1 / norm_img1
    normalized_image2 = img2 / norm_img2

    normalized_image1 = normalized_image1.flatten()
    normalized_image2 = normalized_image2.flatten()

    ncc_score = np.dot(normalized_image1, normalized_image2)
    return ncc_score


def canny_edge_detection(img_2d):
    img_2d = (255 * img_2d).astype(np.uint8)

    img_blur = cv2.GaussianBlur(img_2d, (1, 1), 0)
    edges = cv2.Canny(image=img_blur, threshold1=40, threshold2=300)

    return edges
