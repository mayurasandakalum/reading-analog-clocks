import cv2
import numpy as np

# Load the mask image (result from your earlier processing)
mask = cv2.imread("../data/test_images/test-clock-6.jpg", cv2.IMREAD_GRAYSCALE)

# Desired new height for the display window
new_height = 500  # You can adjust this value as needed

# Calculate the aspect ratio of the mask image
original_height, original_width = mask.shape[:2]
aspect_ratio = original_width / original_height

# Calculate the new width to maintain the aspect ratio
new_width = int(new_height * aspect_ratio)

# Resize the mask image for display
resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Find contours of the hands in the resized mask
contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Load and resize the original image to match the new dimensions
original_img = cv2.imread("../data/test_images/test-clock-6.jpg")
original_img_resized = cv2.resize(
    original_img, (new_width, new_height), interpolation=cv2.INTER_AREA
)

# Iterate over each detected contour
for contour in contours:
    # Find the bounding rectangle of each contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Calculate the center and the angle of the line
    center = tuple(np.int0(rect[0]))
    angle = rect[2]

    # If the rectangle is almost vertical (fix OpenCV angle range issue)
    if angle < -45:
        angle += 90

    # Draw the contour center point
    cv2.circle(original_img_resized, center, 5, (0, 255, 0), -1)

    # Calculate the line endpoints based on angle
    length = 150  # Length of the line (adjust as needed)
    x_end = int(center[0] + length * np.cos(np.radians(angle)))
    y_end = int(center[1] + length * np.sin(np.radians(angle)))

    x_start = int(center[0] - length * np.cos(np.radians(angle)))
    y_start = int(center[1] - length * np.sin(np.radians(angle)))

    # Draw the line
    cv2.line(original_img_resized, (x_start, y_start), (x_end, y_end), (0, 0, 255), 2)

# Show the resulting image with lines drawn
cv2.imshow("Clock Hands", original_img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save the resulting image
cv2.imwrite("../data/clock_hands_with_lines_resized.png", original_img_resized)
