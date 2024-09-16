import numpy as np
import cv2
import os
import ffmpeg
from utils import apply_color_transfer_to_frames, mean_and_std
from color_transfer import ColorTransfer

INPUT_VIDEO = "./data/helloworld_compressed_again.mp4"
INPUT_VIDEO = "./data/bliss-windows-xp-4k-lu_compressed.mp4"
OUTPUT_VIDEO = "./output/output_colorized.mp4"
TARGET_STYLE_IMAGE_PATH = "./data/californicationposter.jpg"

temp_frame_dir = (
    "./temp/"  # Temporary directory forstoring frames during color transfer
)
os.makedirs(temp_frame_dir, exist_ok=True)


if __name__ == "__main__":

    # Extract frames from video
    target_style_img = cv2.imread(TARGET_STYLE_IMAGE_PATH)
    target_style_img = cv2.cvtColor(target_style_img, cv2.COLOR_BGR2LAB)
    ffmpeg.input(INPUT_VIDEO).output(f"{temp_frame_dir}/frame_%04d.png").run()

    # Apply color transfer to each frame
    frame_list = sorted(os.listdir(temp_frame_dir))
    template_mean, template_std = mean_and_std(target_style_img)

    apply_color_transfer_to_frames(
        temp_frame_dir, ColorTransfer.reinhard_transfer, target_style_img
    )

    # Reassemble frames back into a video
    (
        ffmpeg.input(
            f"{temp_frame_dir}/frame_%04d.png", framerate=30
        )  # Assuming 30 fps
        .output(OUTPUT_VIDEO)
        .run()
    )

    # Remove temporary frames in ./temp/
    for frame_name in frame_list:
        os.remove(os.path.join(temp_frame_dir, frame_name))

    print("Color transfer applied to the video and saved as:", OUTPUT_VIDEO)
