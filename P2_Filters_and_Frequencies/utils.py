import numpy as np


class KernelGenerator:
    def __init__(self, kernel_width, kernel_height):
        self.kernel_width = kernel_width
        self.kernel_height = kernel_height

    def dummy_kernel(self):
        kernel = np.ones((self.kernel_width, self.kernel_height))
        return kernel / np.sum(kernel)

    def get_gaussian_kernel(self, sigma):
        kernel = np.zeros((self.kernel_width, self.kernel_height))
        for i in range(self.kernel_width):
            for j in range(self.kernel_height):
                kernel[i, j] = np.exp(
                    -(
                        (i - self.kernel_width // 2) ** 2
                        + (j - self.kernel_height // 2) ** 2
                    )
                    / (2 * sigma**2)
                )
        return kernel / np.sum(kernel)
