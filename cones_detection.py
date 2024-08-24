import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "img.jpg"
image = cv2.imread(image_path)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

'''
color thresholding
'''
# Define the range for the blue color in HSV
lower_blue = np.array([100, 150, 50])  # Lower bound for blue
upper_blue = np.array([130, 255, 255]) # Upper bound for blue
lower_yellow = np.array([20, 150, 100])   # Lower bound for yellow
upper_yellow = np.array([40, 255, 255])   # Upper bound for yellow

# Create a mask for the blue color
blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

# Create a mask for the yellow color
yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)



# Combine the blue and yellow masks
combined_mask = cv2.bitwise_or(blue_mask, yellow_mask)

# Apply the combined mask to the original image
detected_cones = cv2.bitwise_and(image, image, mask=combined_mask)

'''
Opening and closing morphological operations
'''
kernel = np.ones((5, 5), np.uint8)

opened_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

refined_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)

refined_cones = cv2.bitwise_and(image, image, mask=refined_mask)

# Display the original image, the refined mask, and the final result
'''
fig, ax = plt.subplots(2, 3, figsize=(20, 10))


ax[0][0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[0][0].set_title("Original Image")
ax[0][0].axis('off')

ax[0][1].imshow(combined_mask, cmap='gray')
ax[0][1].set_title("Combined Mask")
ax[0][1].axis('off')

ax[0][2].imshow(cv2.cvtColor(detected_cones, cv2.COLOR_BGR2RGB))
ax[0][2].set_title("Detected Blue and Yellow Cones")
ax[0][2].axis('off')

ax[1][0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax[1][0].set_title("Original Image")
ax[1][0].axis('off')

ax[1][1].imshow(refined_mask, cmap='gray')
ax[1][1].set_title("Refined Mask")
ax[1][1].axis('off')

ax[1][2].imshow(cv2.cvtColor(refined_cones, cv2.COLOR_BGR2RGB))
ax[1][2].set_title("Refined Detection of Cones")
ax[1][2].axis('off')

plt.show()
'''


# Find contours in the refined mask
contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw contours on
contour_image = image.copy()

# Filter and draw the contours
for contour in contours:
    # Filter by contour area (you can adjust these thresholds)
    if cv2.contourArea(contour) > 100:  # You might want to experiment with this threshold
        # Draw the contour on the image
        # cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 3)  # Green contours

        # Get the bounding rectangle coordinates
        x, y, w, h = cv2.boundingRect(contour)
        # Draw the rectangle on the image (color: green, thickness: 2)
        cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the original image with contours
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.title("Detected Contours of Cones")
plt.axis('off')
plt.show()
