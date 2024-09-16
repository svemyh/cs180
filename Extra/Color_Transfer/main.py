import numpy as np
import cv2
import os
import ffmpeg
from utils import apply_color_transfer_to_frames, mean_and_std, is_image, is_video
from color_transfer import ColorTransfer

INPUT_FILE = "./data/helloworld_compressed_again.mp4"
INPUT_FILE = "./data/bliss-windowsxp1080p.jpg"
INPUT_FILE = "./data/bliss-windows-xp-4k-lu_compressed.mp4"
INPUT_FILE = "./data/bliss-windowsxp.jpg"

TARGET_STYLE_IMAGE_PATH = "./data/bliss-windowsxp.jpg"
TARGET_STYLE_IMAGE_PATH = "./data/van-gogh-cafe-terrace-at-midnight.jpg"
TARGET_STYLE_IMAGE_PATH = "./data/vangogh-cafe.jpg"
TARGET_STYLE_IMAGE_PATH = "./data/californicationposter.jpg"


OUTPUT_FILE = "./output/output_colorized"
temp_frame_dir = (
    "./temp/"  # Temporary directory forstoring frames during color transfer
)
os.makedirs(temp_frame_dir, exist_ok=True)

if __name__ == "__main__":
    # Load the target style image and convert it to LAB color space
    target_style_img = cv2.imread(TARGET_STYLE_IMAGE_PATH)
    target_style_img = cv2.cvtColor(target_style_img, cv2.COLOR_BGR2LAB)
    # template_mean, template_std = mean_and_std(target_style_img)

    if is_video(INPUT_FILE):
        # Extract frames from the video
        ffmpeg.input(INPUT_FILE).output(f"{temp_frame_dir}/frame_%04d.png").run()

        # Apply color transfer to each frame
        apply_color_transfer_to_frames(
            temp_frame_dir, ColorTransfer.reinhard_transfer, target_style_img
        )

        # Reassemble frames back into a video
        OUTPUT_FILE += ".mp4"
        (
            ffmpeg.input(
                f"{temp_frame_dir}/frame_%04d.png", framerate=30
            )  # Assuming 30 fps
            .output(OUTPUT_FILE)
            .run()
        )

        # Remove temporary frames
        for frame_name in os.listdir(temp_frame_dir):
            os.remove(os.path.join(temp_frame_dir, frame_name))
        os.rmdir(temp_frame_dir)  # Remove the temp directory

        print("Color transfer applied to the video and saved as:", OUTPUT_FILE)

    elif is_image(INPUT_FILE):
        # Process the image
        input_img = cv2.imread(INPUT_FILE)
        colorized_img = ColorTransfer.reinhard_transfer(input_img, target_style_img)
        OUTPUT_FILE += ".jpg"
        cv2.imwrite(OUTPUT_FILE, colorized_img)

        print("Color transfer applied to the image and saved as:", OUTPUT_FILE)

    else:
        print("Unsupported file type. Please provide an image or video file.")
