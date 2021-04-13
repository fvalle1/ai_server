import numpy as np
import cv2
import cv2 as cv
import time


# capture camera
cap = cv2.VideoCapture(0)

# Read an image.
#frame = cv.imread('input.jpg')
#if frame is None:
#    raise Exception('Image not found!')

ret, frame = cap.read()
# Display the resulting frame
#Save the frame to an image file.
cv.imwrite('input.jpg', frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
