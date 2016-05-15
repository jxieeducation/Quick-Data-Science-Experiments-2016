from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import sys

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def getWidth(loc):
	image = cv2.imread(loc)
	image = imutils.resize(image, width=min(500, image.shape[1]))

	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
	if rects is None:
		return -1

	maxArea = -1
	myList = ()
	for (x, y, w, h) in rects:
		if w * h > maxArea:
			maxArea = w * h
			myList = (x, y, w, h)
	x, y, w, h = myList	

	# this is generated
	person_image = image[y:y+h, x:x+w]

	# this is box
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.imshow("Body recogniton", image)
	cv2.waitKey(0)

loc = sys.argv[1] 
print getWidth(loc)
