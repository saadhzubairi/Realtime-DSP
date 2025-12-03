import cv2
import numpy as np 
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
imagefile = os.path.join(base_dir, 'fruit.jpg')

img = cv2.imread(imagefile, 1)   

# Convert to different color space
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

print(type(img_hsv))
print(img_hsv.shape)
print(img_hsv.dtype)

blue = np.uint8([[[0, 255, 0]]])   # 3D array describing green in BGR
blue_hsv = cv2.cvtColor(blue, cv2.COLOR_BGR2HSV)
h = blue_hsv[0,0,0]
print('Blue in HSV color space:', blue_hsv)
print('Hue = ', h)   # see that h = 120

lower = np.array([max(h-20, 0), 50, 50])
upper = np.array([min(h+20, 179), 255, 255])
print('lower = ', lower)
print('upper = ', upper)

# quit()

# Determine binary mask
blue_mask = cv2.inRange(img_hsv, lower, upper)

# Apply mask to color image
output = cv2.bitwise_and(img, img, mask = blue_mask)

# Show images:
cv2.imshow('Original image', img)
cv2.imshow('Mask', blue_mask)
cv2.imshow('Segmented image', output)

print('Switch to images. Then press any key to stop')

cv2.waitKey(0)
cv2.destroyAllWindows()

# Write the image to a file
cv2.imwrite('fruit_mask_green.jpg', blue_mask)   
cv2.imwrite('fruit_green.jpg', output)   