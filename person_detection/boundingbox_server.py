from flask import Flask, render_template, request
from werkzeug import secure_filename
app = Flask(__name__)

from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import sys
import json

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		f = request.files['file']
		newFileName = "test." + f.filename.rsplit(".")[-1]
		f.save(newFileName) # we want to keep the file extension
		# image recognition stuff here
		image = cv2.imread(newFileName)
		image = imutils.resize(image, width=min(500, image.shape[1]))
		orig = image.copy()
		(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
			padding=(8, 8), scale=1.05)
		for (x, y, w, h) in rects:
			cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
		rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
		pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
		myBoundingBox = {}
		myBoundingBox['xA'], myBoundingBox['yA'], myBoundingBox['xB'], myBoundingBox['yB'] = pick[0]
		return json.dumps(myBoundingBox)

if __name__ == '__main__':
   app.run(debug = True)
