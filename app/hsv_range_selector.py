import cv2
import numpy as np


def nothing(x):
    pass


# List of image paths
image_paths = [
    "../data/test_images/test-clock-6.jpg",
    "../data/test_images/test-clock-7.jpg",
    "../data/test_images/captured-images/20241102_140018.jpg",
]

# Desired new height
new_height = 300  # Adjust the height as needed


# Function to resize images
def resize_image(image, new_height):
    height, width = image.shape[:2]
    new_width = int(width * new_height / height)
    return cv2.resize(image, (new_width, new_height))


# Load and resize images
images = []
for path in image_paths:
    img = cv2.imread(path)
    if img is not None:
        img = resize_image(img, new_height)
        images.append(img)
    else:
        print(f"Failed to load image at path: {path}")

# Check if images list is not empty
if not images:
    print("No images to display.")
    exit()

# Create a window
cv2.namedWindow("image")

# Create trackbars for color change
cv2.createTrackbar("HMin", "image", 0, 179, nothing)
cv2.createTrackbar("SMin", "image", 0, 255, nothing)
cv2.createTrackbar("VMin", "image", 0, 255, nothing)
cv2.createTrackbar("HMax", "image", 0, 179, nothing)
cv2.createTrackbar("SMax", "image", 0, 255, nothing)
cv2.createTrackbar("VMax", "image", 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos("HMax", "image", 179)
cv2.setTrackbarPos("SMax", "image", 255)
cv2.setTrackbarPos("VMax", "image", 255)

# Initialize variables to store previous trackbar values
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while True:
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos("HMin", "image")
    sMin = cv2.getTrackbarPos("SMin", "image")
    vMin = cv2.getTrackbarPos("VMin", "image")
    hMax = cv2.getTrackbarPos("HMax", "image")
    sMax = cv2.getTrackbarPos("SMax", "image")
    vMax = cv2.getTrackbarPos("VMax", "image")

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Initialize list to store processed images
    results = []

    # Process each image
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)
        results.append(result)

    # Concatenate images horizontally
    combined_result = np.hstack(results)

    # Print if there is a change in HSV value
    if (
        (phMin != hMin)
        or (psMin != sMin)
        or (pvMin != vMin)
        or (phMax != hMax)
        or (psMax != sMax)
        or (pvMax != vMax)
    ):
        print(
            f"(hMin = {hMin}, sMin = {sMin}, vMin = {vMin}), (hMax = {hMax}, sMax = {sMax}, vMax = {vMax})"
        )
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display combined image
    cv2.imshow("image", combined_result)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
