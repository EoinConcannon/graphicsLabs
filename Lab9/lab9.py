# imports
import cv2
import numpy as np
from matplotlib import pyplot as plt

# ATU1.jpg
# ATU2.jpg
# rome.jpg
# tokyo.jpg
img = cv2.imread('ATU1.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # prevent colour issues
img2 = cv2.imread('ATU2.jpg')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
greyImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converts to greyscale

nrows = 3
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

# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
imgBruteForce = cv2.drawMatches(img,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.subplot(nrows, ncols,5),plt.imshow(imgBruteForce, cmap = 'gray') # adding BruteForceMatcher image to image plot
plt.title('BruteForceMatcher'), plt.xticks([]), plt.yticks([])

# FLANN
# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)

imgFLANN = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.subplot(nrows, ncols,6),plt.imshow(imgFLANN, cmap = 'gray')
plt.title('FLANN'), plt.xticks([]), plt.yticks([])

plt.show() # display images