import numpy as np
import matplotlib.pyplot as plt
from utils import translate_image


def test_translate_image():
    # Create a simple 5x5 test image with a white square in the center
    img = np.zeros((5, 5), dtype=np.uint8)
    img[2, 2] = 255  # Add a single white pixel at the center

    # Test translation with various x and y offsets
    translations = [
        (1, 0),  # Translate right by 1 column (x_offset positive)
        (-1, 0),  # Translate left by 1 column (x_offset negative)
        (0, 1),  # Translate upwards by 1 row (y_offset positive)
        (0, -1),  # Translate downwards by 1 row (y_offset negative)
        (1, 1),  # Translate right by 1 column and upwards by 1 row
    ]

    # Plot original and translated images
    plt.figure(figsize=(10, 5))

    for i, (x_offset, y_offset) in enumerate(translations):
        translated_img = translate_image(img, x_offset, y_offset)

        # Plot the result
        plt.subplot(2, 3, i + 2)  # Start from 2 to leave space for original image
        plt.imshow(translated_img, cmap="gray")
        plt.title(f"Offset (x={x_offset}, y={y_offset})")
        plt.axis("off")

    # Plot the original image for comparison
    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# Run the test
test_translate_image()
