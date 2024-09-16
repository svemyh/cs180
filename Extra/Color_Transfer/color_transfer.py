import cv2
import numpy as np
from utils import mean_and_std


class ColorTransfer:
    @staticmethod
    def identity_transfer(input_frame, target_style_img):
        """ "For testing purposes. Returns the input frame as is."""
        return input_frame

    @staticmethod
    def reinhard_transfer(input_frame, target_style_img, alpha=1.05):
        """Apply Color Transfer [Reinhard, et al, 2001] to the input frame.

        Reference: Erik Reinhard, Michael Ashikhmin, Bruce Gooch, Peter Shirley, Color Transfer between
        Images. IEEE Computer Graphics and Applications, 21(5), pp. 34–41. September 2001.

        Inspired by @DigitalSreeni. https://youtu.be/_GAhbrGHaVo?si=SQE5u97FHo_kStPR.
        """

        img = cv2.cvtColor(input_frame, cv2.COLOR_BGR2LAB)
        target_style_img = cv2.cvtColor(input_frame, cv2.COLOR_BGR2LAB)

        _, _, colour_channels = img.shape
        for chan in range(colour_channels):
            target_style_img_mean, target_style_img_std = mean_and_std(
                target_style_img[:, :, chan]
            )
            img_mean, img_std = mean_and_std(img[:, :, chan])

            img[:, :, chan] = (
                (img[:, :, chan] - img_mean) * (target_style_img_std / img_std)
            ) + target_style_img_mean * alpha

        reinhard_img = np.clip(np.round(img), 0, 255).astype(np.uint8)

        return cv2.cvtColor(reinhard_img, cv2.COLOR_LAB2BGR)

    @staticmethod
    def clip(x, min_value=0, max_value=255):
        """Clips values x to be within the specified min and max range."""
        return max(min_value, min(x, max_value))
