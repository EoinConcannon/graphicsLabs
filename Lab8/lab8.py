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
ncols = 2

plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

KernelSizeWidth = 3
KernelSizeHeight = 3

img = cv2.GaussianBlur(img,(KernelSizeWidth, KernelSizeHeight),0)


plt.subplot(nrows, ncols,3),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap = 'gray')
plt.title('3x3 Blur'), plt.xticks([]), plt.yticks([])

KernelSizeWidth = 13
KernelSizeHeight = 13

img = cv2.GaussianBlur(img,(KernelSizeWidth, KernelSizeHeight),0)

plt.subplot(nrows, ncols,4),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap = 'gray')
plt.title('13x13 Blur'), plt.xticks([]), plt.yticks([])
plt.show()