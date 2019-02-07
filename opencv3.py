#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 23:38:44 2018

@author: aman
"""


import cv2
import numpy as np
##########################   COLOR SPACE HSV   ##############################
# HSV is particularly useful for color filtering.
image = cv2.imread('./images/input.jpg')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# This converts the image to HSV format, i.e. Hue, Saturation and Value. Hue denotes the colors, saturation denotes the whiteness or vividness of the colors, and the value denotes the lightness or darkness, the intensoty of the image.



########################## NOW, the array of hsv_image are the lenghth, breadth and the channel, 0 for hue, 1 for saturation and 2 for the value channel. hsv_image[:, :, 1] Means we are selecting all of the length and the breadth and then the saturation channel.#############################3



cv2.imshow('HSV Image', hsv_image)
# This is used to display the hsv image as it is, looks horrible. owing to the fact that the image is run in hsv mode but is still displayed in the RGB mode, and the HSV mode has only 180 values, that is less colors.

cv2.imshow('Hue Channel', hsv_image[:, :, 0])
# This has the third attribute as zero, that is here we are viewing the hue image.

cv2.imshow('Saturation Channel', hsv_image[:, :, 1])
# This has the third attribute as 1, that is the saturation channel.

cv2.imshow('Value Channel', hsv_image[:, :, 2])
# The intensity or value or brightness channel.

cv2.waitKey()



cv2.destroyAllWindows()


wait = True
while wait:
    wait = cv2.waitKey()=='q113'

