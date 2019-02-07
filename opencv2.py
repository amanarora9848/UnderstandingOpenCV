#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 23:26:11 2018

@author: aman
"""

import cv2
import numpy as np
image = cv2.imread("./images/input.jpg")

#We are accessing the three values stored in the first pixel location.
B, G, R = image[0, 0]
print(B, G, R)
img_array = np.shape(image)
print(img_array)

# Now convert image to grayscale.
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img_array2 = np.shape(gray_image)
print(img_array2)
# Now we see that only 2 elements are returned in the array. 

# Now, if we want to view the BGR values,
print(gray_image[0, 0])
# Now, we actually see only one value, in gray_scale being printed, instead of three.