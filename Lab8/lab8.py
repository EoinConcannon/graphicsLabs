# imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg')

# https://stackoverflow.com/questions/55686826/how-can-i-properly-convert-image-to-gray-scale-in-opencv
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converts to greyscale
# cv2.imshow('image', img)
# cv2.waitKey()

nrows = 2
ncols = 1

plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.show()