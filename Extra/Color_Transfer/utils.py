import os
import cv2
import numpy as np


def mean_and_std(image):
    mean, std = cv2.meanStdDev(image)
    mean = np.round(mean.flatten(), 2)
    std = np.round(std.flatten(), 2)
    return mean, std


def apply_color_transfer_to_frames(
    temp_frame_dir, apply_color_transfer, target_style_img
):
    """Apply color transfer to all frames in the specified directory."""

    frame_list = sorted(os.listdir(temp_frame_dir))
    for frame_name in frame_list:
        frame_path = os.path.join(temp_frame_dir, frame_name)
        frame = cv2.imread(frame_path)

        modified_frame = apply_color_transfer(frame, target_style_img)

        cv2.imwrite(frame_path, modified_frame)
