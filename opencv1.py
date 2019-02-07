#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 21:58:21 2018

@author: aman
"""

import numpy as np
import cv2
input = cv2.imread("./images/input.jpg")
#Loads an image, the arguement stores the path.


cv2.imshow("Hello world", input)
#To display the image. Has two arguements, name of image, and the image file that we wanna open.

cv2.waitKey()
#Not necessary, waits for you to press any key.


########GREY-SCALING AN IMAGE IN OPENCV, THE cv2.cvtColor() function#########
gray_image = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
#This takes the image and passs cv2.COLOR+BGR2GRAY to convert image to grayscale image.
cv2.imshow("GrayScaled", gray_image)
cv2.waitKey()


#Now another thing that we can do is : 

# Directly, put another arguement in the imread() function, like :-
# cv2.imread("./images/input.jpg", 0)
# The arguement 0 converts the image to grayscale in the step itself to save coding time, but the processing time is same.



cv2.destroyAllWindows()
#Very important, otherwise, window will crash.

wait = True
while wait:
  wait = cv2.waitKey()=='q113' # hit q to exit

#We use numpy and use .shape() function of numpy, to look at dimensions of the array, in the form of array itself.
img_array = np.shape(input)
print(img_array)
print("Height in pixels: ", img_array[0])
print("Width in pixels: ", img_array[1])

#How do we write images in openCV
cv2.imwrite("output.png", input)
#First arg : name of the file we are saving as.
#Second arg is the file_name.
#Can store in multiple formats.
#Written into the working directory.