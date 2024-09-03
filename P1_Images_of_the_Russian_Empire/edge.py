import cv2
import numpy as np


def sobel_metric(img1, img2, dx, dy):
    # skio.imsave("tmp/img1.jpg", img1)
    # skio.imsave("tmp/img2.jpg", img2)

    img1 = cv2.imread("tmp/img1.jpg")
    img2 = cv2.imread("tmp/img2.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1_blur = cv2.GaussianBlur(img1, (5, 5), 0)
    img2_blur = cv2.GaussianBlur(img2, (5, 5), 0)
    img1_sobel = cv2.Sobel(
        src=img1_blur, ddepth=cv2.CV_64F, dx=dx, dy=dy, ksize=5
    )  # Sobel Edge Detection
    img2_sobel = cv2.Sobel(src=img2_blur, ddepth=cv2.CV_64F, dx=dx, dy=dy, ksize=5)
    img1_sobel = np.uint8(np.absolute(img1_sobel))
    img2_sobel = np.uint8(np.absolute(img2_sobel))

    # return 0.5*abs(mse(img1_sobel, img2_sobel)) + 0.5 * abs(ncc2(img1_sobel, img2_sobel)) * 0.5*ssim_metric(img1_sobel, img2_sobel)
    return 1


if __name__ == "__main__":
    # Read the original image
    img = cv2.imread("data/monastery.jpg")
    # Display original image
    cv2.imshow("Original", img)
    cv2.waitKey(0)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(
        src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5
    )  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(
        src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5
    )  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(
        src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5
    )  # Combined X and Y Sobel Edge Detection
    # Display Sobel Edge Detection Images
    cv2.imshow("Sobel X", sobelx)
    cv2.waitKey(0)
    cv2.imshow("Sobel Y", sobely)
    cv2.waitKey(0)
    cv2.imshow("Sobel X Y using Sobel() function", sobelxy)
    cv2.waitKey(0)

    # Canny Edge Detection
    edges = cv2.Canny(
        image=img_blur, threshold1=100, threshold2=200
    )  # Canny Edge Detection
    # Display Canny Edge Detection Image
    cv2.imshow("Canny Edge Detection", edges)
