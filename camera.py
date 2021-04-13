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

while(True):    # Capture frame-by-frame
	ret, frame = cap.read()
	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# Save the frame to an image file.
cv.imwrite('out.png', frame)


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
