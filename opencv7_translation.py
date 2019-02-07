#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 18:55:11 2019

@author: aman
"""

#################    IMAGE MANIPULATION IN USING OPENCV   ##################
# Translation - moving an image in some direction.
# Translation matrix

import cv2
import numpy as np

image = cv2.imread('./images/input.jpg')

# Store the height and width of the image.
height, width = image.shape[:2]

quarter_height, quarter_width = height/4, width/4

# T = /1 0 Tx/
#    /0 1 Ty/, T being the translation matrix.

# Now we use numpy, see below.
T = np.float32([[1, 0, quarter_height], [0, 1, quarter_width]])

# Now, we use the warpAffine to transform the image using the matrix, T.
img_translation = cv2.warpAffine(image, T, (width, height))
cv2.imshow('Translation of the image.', img_translation) 
#The translated image is shown, shifted both horizontally and vertically by 1/4th of the respective dimensions.

cv2.waitKey(0)
cv2.destroyAllWindows()