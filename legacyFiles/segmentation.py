import cv2
import numpy as np

def resize_to_fit_screen(image, max_width=800, max_height=600):
    height, width = image.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def segment_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Histogram equalization
    equalized = cv2.equalizeHist(gray)

    # Gaussian blur
    blurred = cv2.GaussianBlur(equalized, (7, 7), 0)

    # Adaptive thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morphed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)

    # Contouring
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(sorted_contours) >= 2:
        second_largest_item = sorted_contours[1]

        # Find the minimum bounding rectangle
        rect = cv2.minAreaRect(second_largest_item)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Create a mask for the minimum bounding rectangle
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [box], -1, 255, -1)

        # Apply the mask to isolate the package
        masked_image = cv2.bitwise_and(frame, frame, mask=mask)
        return masked_image, box

    return None, None
