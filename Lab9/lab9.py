# imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ATU1.jpg
# ATU2.jpg
# rome.jpg
# tokyo.jpg
img = cv2.imread('ATU2.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # prevent colour issues
greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converts to greyscale

nrows = 2
ncols = 2

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

# Shi Tomasi algorithm (GFTT "Good Features To Track")
maxCorners = 250 # use different numbers
qualityLevel = 0.01
minDistance = 10

corners = cv2.goodFeaturesToTrack(greyImg,maxCorners,qualityLevel,minDistance)

imgShiTomasi = img.copy() # deep copy

for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi,(x,y),3,(256, 0, 256),-1)

# https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
imgORB = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)

plt.subplot(nrows, ncols,1),plt.imshow(img, cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,2),plt.imshow(imgHarris, cmap = 'gray')
plt.title('Harris'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,3),plt.imshow(imgShiTomasi, cmap = 'gray')
plt.title('GFTT'), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols,4),plt.imshow(imgORB, cmap = 'gray')
plt.title('ORB'), plt.xticks([]), plt.yticks([])

plt.show() # display images