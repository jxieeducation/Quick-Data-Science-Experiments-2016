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

	maxArea = -1
	myList = ()
	for (x, y, w, h) in rects:
		if w * h > maxArea:
			maxArea = w * h
			myList = (x, y, w, h)
	if len(myList) < 1:
		return -1
	x, y, w, h = myList
	myWidth = w

	# this is generated
	person_image = image[y:y+h/2, x:x+w]

	# this is box
	image = image.copy()
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.imshow("Body recogniton", image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	####### card recog stuff now #######
	gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 50, 17, 17)
	edged = cv2.Canny(gray, 30, 200)
	(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	screenCnt = None

	approx = None
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.1 * peri, True)
		if len(approx) == 4:
			tophalf = person_image.copy()
			cv2.drawContours(tophalf, [approx], -1, (0, 255, 255), 3)
			cv2.imshow("Game Boy Screen", tophalf)
			cv2.waitKey(0)
			break
	if approx is None:
		return -1

	card_pixels = list(approx[2][0])[0] - list(approx[0][0])[0]
	print "card pixels: %d" % card_pixels
	print "person width: %d" % myWidth
	shoulderWidthUpper = float(myWidth)/ 2.5 / card_pixels * 8.5
	shoulderWidthLower = float(myWidth)/ 2.5 / card_pixels * 8
	print "shoulder width: %3f ~ %3f " % (shoulderWidthLower, shoulderWidthUpper)
	return (shoulderWidthLower, shoulderWidthUpper)

loc = sys.argv[1] 
print getWidth(loc)
