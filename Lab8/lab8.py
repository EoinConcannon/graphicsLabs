# imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg')

cv2.imshow('image', img)
cv2.waitKey()

# https://stackoverflow.com/questions/55686826/how-can-i-properly-convert-image-to-gray-scale-in-opencv