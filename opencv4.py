#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 01:13:49 2018

@author: aman
"""

import cv2
import numpy as np

image = cv2.imread('./images/input.jpg')

# OpenCV's split() splits the image to individual colors index.
B, G, R = cv2.split(image)
bShape = np.shape(G)
print(bShape)
cv2.imshow("Red", R)
cv2.imshow('Blue', B)
cv2.imshow('Green', G)
# Here we see the Grayscale values corresponding to each BGR.

#####    MERGING THE BGR VALUES TO RECREATE THE ACTUAL IMAGE   ######

# This is achieved using the merged function of cv2, ie, cv2.merge()
merged = cv2.merge([B, G, R])
cv2.imshow("MergedImage", merged)


#########     COLOR AMPLIFICATION       #########

# We can also add extra color to an existing color out og B, G and R, though it can go to maximum value of 255, even if the sum is greater than that.

merged = cv2.merge([B+100, G+100, R+100])
cv2.imshow("All of them amplified by 100", merged)




########################################################################

# What if we wanted to see Red Blue Green actual colors as opposed to greyscale images?
# Let's create a matrix called zeros, so we craete array of zeros, that are the same shape of the original image.

zeros = np.zeros(image.shape[:2], dtype = 'uint8')# Array of 0s which are of same shape as original image
cv2.imshow("Red image", cv2.merge([zeros, zeros, R]))
cv2.imshow("Green image", cv2.merge([zeros, G, zeros]))
cv2.imshow('Blue image', cv2.merge([B, zeros, zeros]))

#See the output of the below code snippet.
#image.shape[:2]
#Out[9]: (830, 1245)

################### This is a beautiful experience. #######################

cv2.waitKey(0)


cv2.destroyAllWindows()


wait = True
while wait:
    wait = cv2.waitKey()=='q113'