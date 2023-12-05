# imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('ATU.jpg')

nrows = 2
ncols = 2

# no blur original & greyscale
plt.subplot(nrows, ncols,1),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

# 3x3 blur set
KernelSizeWidth = 3
KernelSizeHeight = 3
img = cv2.GaussianBlur(img,(KernelSizeWidth, KernelSizeHeight),0)

# sobel edge detection vertical
sobelVertical = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5) # y dir
verticalImg = cv2.convertScaleAbs(sobelVertical)

plt.subplot(nrows, ncols,3),plt.imshow(cv2.cvtColor(verticalImg, cv2.COLOR_BGR2GRAY), cmap = 'gray')
plt.title('3x3 Blur'), plt.xticks([]), plt.yticks([])

# 13x13 blur set
KernelSizeWidth = 13
KernelSizeHeight = 13
img = cv2.GaussianBlur(img,(KernelSizeWidth, KernelSizeHeight),0)

# sobel edge detection horizontal
sobelHorizontal = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5) # x dir
horizontalImg = cv2.convertScaleAbs(sobelHorizontal)

plt.subplot(nrows, ncols,4),plt.imshow(cv2.cvtColor(horizontalImg, cv2.COLOR_BGR2GRAY), cmap = 'gray')
plt.title('13x13 Blur'), plt.xticks([]), plt.yticks([])
plt.show()

