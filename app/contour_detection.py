import cv2
import numpy as np


def nothing(x):
    pass


def main(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image.")
        return

    # Get original dimensions
    height, width = img.shape[:2]
    # Set desired height
    desired_height = 480  # Adjust this value as needed
    # Calculate the scaling factor
    scale_ratio = desired_height / height
    # Calculate new width to maintain aspect ratio
    new_width = int(width * scale_ratio)
    # Resize the image
    img = cv2.resize(img, (new_width, desired_height))

    cv2.namedWindow("Contour Detection")

    cv2.createTrackbar("Threshold1", "Contour Detection", 100, 500, nothing)
    cv2.createTrackbar("Threshold2", "Contour Detection", 200, 500, nothing)
    cv2.createTrackbar("Min Area", "Contour Detection", 1000, 100000, nothing)
    cv2.createTrackbar(
        "Blur", "Contour Detection", 1, 31, nothing
    )  # New trackbar for blur

    while True:
        thresh1 = cv2.getTrackbarPos("Threshold1", "Contour Detection")
        thresh2 = cv2.getTrackbarPos("Threshold2", "Contour Detection")
        min_area = cv2.getTrackbarPos("Min Area", "Contour Detection")
        blur_ksize = cv2.getTrackbarPos("Blur", "Contour Detection")

        # Ensure the blur kernel size is odd and greater than or equal to 1
        if blur_ksize % 2 == 0:
            blur_ksize += 1
        if blur_ksize < 1:
            blur_ksize = 1

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, thresh1, thresh2)

        contours, hierarchy = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        img_contours = img.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                cv2.drawContours(img_contours, [cnt], -1, (0, 255, 0), 2)

        cv2.imshow("Contour Detection", img_contours)
        cv2.imshow("Edges", edges)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "../data/test_images/test-clock-1.jpg"  # Replace with your image path
    main(image_path)
