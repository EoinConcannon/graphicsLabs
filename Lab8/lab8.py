# imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg')

plt.plot([1,2,3])
# now create a subplot which represents the top plot of a grid
# with 2 rows and 1 column. Since this subplot will overlap the
# first, the plot (and its axes) previously created, will be removed
plt.subplot(211)
plt.plot(range(12))
plt.subplot(212, facecolor='y') # creates 2nd subplot with yellow background

# https://stackoverflow.com/questions/55686826/how-can-i-properly-convert-image-to-gray-scale-in-opencv
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converts to greyscale
# cv2.imshow('image', img)
# cv2.waitKey()

plt.subplot(nrows, ncols,1),plt.imshow(imgOrig, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(cv2.cvtColor(imgOrig, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
plt.show()