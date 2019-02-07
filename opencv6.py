# Drawing images and shapes using openCV.

import cv2
import numpy as np

# Create a black image.
image = np.zeros((512, 512, 3), np.uint8)

# Create a greyscale image, just remove three 3 (dimension) there, looks identcal.
#image_bw = np.zeros((512, 512), np.uint8)
#
#cv2.imshow('Black rectangle (Color) ', image)
#cv2.imshow('Black rectangle (B&W)', image_bw)
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#Drawing a line over our black square.
cv2.line(image, (0, 0), (80, 80), (55, 200, 160), 5)



#Drawing a Rectangle.
#cv2.rectangle(image, (90, 90), (290, 290), (280, 280, 50), 5)
cv2.rectangle(image, (90, 90), (290, 290), (280, 280, 50), -1)


#Creating circles.
cv2.circle(image, (340, 340), 60, (215, 75, 100), 4)

#Creating polygons.

#Defining the points.
pts = np.array([[400, 400], [480, 400], [480, 500]], np.int32)
cv2.polylines(image, [pts], True, (0, 0, 255), 2)
cv2.imshow('Different shapes and sizes offered by openCV', image)


cv2.waitKey(0)
cv2.destroyAllWindows()