import imutils
from skimage import exposure
import numpy as np
import argparse
import cv2
import sys
 
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-q", "--query", required = True,
#     help = "Path to the query image")
# args = vars(ap.parse_args())

# load the query image, compute the ratio of the old height
# to the new height, clone it, and resize it
image = cv2.imread(sys.argv[1])
ratio = image.shape[0] / 300.0
orig = image.copy()
image = imutils.resize(image, height = 300)

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 50, 17, 17)
edged = cv2.Canny(gray, 30, 200)

# find contours in the edged image, keep only the largest
# ones, and initialize our screen contour
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None
#print cnts
mycopy = image.copy()

# loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    image = mycopy.copy()
    cv2.drawContours(image, [approx], -1, (0, 255, 255), 3)
    cv2.imshow("Game Boy Screen", image)
    cv2.waitKey(0)
