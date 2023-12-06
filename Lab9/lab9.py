# imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU1.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # prevent colour issues

greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converts to greyscale

cv2.imshow('image', img)
cv2.waitKey()