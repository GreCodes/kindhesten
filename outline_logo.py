import cv2
import numpy as np

# Load image
img = cv2.imread('logo.png', cv2.IMREAD_GRAYSCALE)

# Threshold to isolate dark lines
_, thresh = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY_INV)

# Find contours (outlines)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create blank canvas (white, with transparency)
outline = np.ones((img.shape[0], img.shape[1], 4), dtype=np.uint8) * 255

# Draw contours in black
cv2.drawContours(outline, contours, -1, (0,0,0,255), 2)

# Save as PNG
cv2.imwrite('logo-outline.png', outline)