import glob
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image


def show(img):
	temp = np.swapaxes(img, 0, 2)
	fixed_img = np.swapaxes(temp, 0, 1)
	plt.imshow(fixed_img)
	plt.show()


def readImage(mydir, width=32):
	img = Image.open(mydir).convert('RGB')
	img = img.resize((width, width), PIL.Image.ANTIALIAS)
	return np.array(img).reshape((width, width, 3))


def loadDataset():
	# getting the ratings
	ratingDirs = glob.glob("./hot_or_not_image_and_rating_data/female/*.txt")
	ratingDirs += glob.glob("./hot_or_not_image_and_rating_data/male/*.txt")
	ratings = np.array([float(open(myDir).read().split('\n')[0]) for myDir in ratingDirs])

	imgDirs = glob.glob("./hot_or_not_image_and_rating_data/female/*.jpg")
	imgDirs += glob.glob("./hot_or_not_image_and_rating_data/male/*.jpg")
	images = np.array([readImage(myDir) for myDir in imgDirs])
	images = np.swapaxes(np.swapaxes(images, 2, 3), 1, 2)
	images = images.astype('float32')
	images /= 255
	return images, ratings
