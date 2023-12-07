# imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # prevent colour issues

greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converts to greyscale

# Harris corner detection
blockSize = 2
aperture_size = 3
k = 0.04

dst = cv2.cornerHarris(greyImg, blockSize, aperture_size, k)

# https://stackoverflow.com/questions/16533078/clone-an-image-in-cv2-python
imgHarris = img.copy() # deep copy

threshold = 0.1; #number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris,(j,i),3,(256, 0, 256),-1)

cv2.imshow('image', imgHarris)
cv2.waitKey()