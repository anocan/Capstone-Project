import cv2 as cv
import numpy as np

# Function to rescale the frame dynamically for a screen-friendly size
def resize_to_fit_screen(image, max_width=800, max_height=600):
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return cv.resize(image, (new_width, new_height), interpolation=cv.INTER_AREA)

# Load and preprocess the image
img = cv.imread('/Users/anilbudak/VSCode/Bitirme/CNN/testData/testVIDEO.png')

# Convert to grayscale and apply GaussianBlur
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray, (1, 1), 0)

# Use Canny edge detection
edges = cv.Canny(blur, 40, 100)

# Morphological operations to connect edges
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
morphed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)

# Find contours
contours, _ = cv.findContours(morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Identify the package-like contour
package_contour = None
max_score = 0
for contour in contours:
    epsilon = 0.02 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    
    # Check if the shape is "rectangle-like"
    if 4 <= len(approx) <= 8:  # Flexible number of corners
        area = cv.contourArea(contour)
        x, y, w, h = cv.boundingRect(contour)
        aspect_ratio = w / float(h)
        extent = area / (w * h)  # Ratio of contour area to bounding rectangle area
        
        # Score based on how rectangle-like the contour is
        score = (0.5 * extent) + (0.5 * (0.7 <= aspect_ratio <= 1.8))  # Weighted score
        
        if score > max_score and area > 800:  # Adjust area threshold as needed
            max_score = score
            package_contour = contour

# Create a mask for the detected package
mask = np.zeros(img.shape[:2], dtype='uint8')
if package_contour is not None:
    cv.drawContours(mask, [package_contour], -1, 255, -1)  # Fill the package area with white
    masked_image = cv.bitwise_and(img, img, mask=mask)

    # Invert mask to black out surroundings
    surroundings_mask = cv.bitwise_not(mask)
    surroundings_blacked = cv.bitwise_and(img, img, mask=surroundings_mask)

    # Show results
    cv.imshow('Original Image', resize_to_fit_screen(img))
    cv.imshow('Package Detected', resize_to_fit_screen(masked_image))
    cv.imshow("Canny Edges", resize_to_fit_screen(edges))
    cv.imshow('Blacked Surroundings', resize_to_fit_screen(surroundings_blacked))
    
else:
    print("No rectangle-shaped package found.")

cv.waitKey(0)