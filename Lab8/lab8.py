# imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ATU.jpg
# rome.jpg
# tokyo.jpg
img = cv2.imread('tokyo.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # prevent colour issues

nrows = 2
ncols = 3

greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converts to greyscale

# no blur original & greyscale
plt.subplot(nrows, ncols,1),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(greyImg, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])

# 3x3 blur set
KernelSizeWidth = 3
KernelSizeHeight = 3
imgBlur = cv2.GaussianBlur(greyImg,(KernelSizeWidth, KernelSizeHeight),0)
# sobel edge detection vertical
# https://stackoverflow.com/questions/51167768/sobel-edge-detection-using-opencv
sobelVertical = cv2.Sobel(imgBlur,cv2.CV_64F,0,1,ksize=5) # y dir
verticalImg = cv2.convertScaleAbs(sobelVertical)
# end result img
plt.subplot(nrows, ncols,3),plt.imshow(sobelVertical, cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

# 13x13 blur set
KernelSizeWidth = 13
KernelSizeHeight = 13
imgBlurStrong = cv2.GaussianBlur(greyImg,(KernelSizeWidth, KernelSizeHeight),0)
# sobel edge detection horizontal
sobelHorizontal = cv2.Sobel(imgBlurStrong,cv2.CV_64F,1,0,ksize=5) # x dir
horizontalImg = cv2.convertScaleAbs(sobelHorizontal)
# end result img
plt.subplot(nrows, ncols,4),plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

# combines previous x/y sobel imgs
sobelCombined = cv2.addWeighted(sobelVertical, 0.5, sobelHorizontal, 0.5, 0)

plt.subplot(nrows, ncols,5),plt.imshow(sobelCombined, cmap = 'gray')
plt.title('Sobel Sum'), plt.xticks([]), plt.yticks([])

# canny img
cannyThreshold = 80
cannyParam2 = 200

canny = cv2.Canny(imgBlur,cannyThreshold,cannyParam2)

plt.subplot(nrows, ncols,6),plt.imshow(canny, cmap = 'gray')
plt.title('Canny Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()