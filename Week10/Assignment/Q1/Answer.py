import numpy as np 
import cv2

cap = cv2.VideoCapture(0)

print("Switch to video window. Then press 'p' to save image, 'q' to quit")

# High-pass kernel highlights edges instead of smoothing
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]])

while True:

    [ok, frame] = cap.read()          # Read one frame

    # Use 2D filtering with the high-pass kernel to emphasize edges
    frame = cv2.filter2D(frame, -1, kernel)

    cv2.imshow('Live video (edges)', frame)

    key = cv2.waitKey(1)
    # key = key & 0xFF      # (May not be necessary)

    if key == ord('p'):
        cv2.imwrite('edges.jpg', frame)              
        
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
