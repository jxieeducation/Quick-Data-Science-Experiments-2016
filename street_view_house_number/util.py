import matplotlib.pyplot as plt
import numpy as np

def show(img):
	# is (3, 32, 32)
	# want (32, 32, 3)
	temp = np.swapaxes(img, 0, 2)
	fixed_img = np.swapaxes(temp, 0, 1)
	plt.imshow(fixed_img)
	plt.show()


def reshapeToKeras2D(matrix):
	# is (32, 32, 3, 73257)
	# want (73257, 3, 32, 32)
	temp1 = np.swapaxes(matrix, 1, 3) # (32, 73257, 3, 32)
	temp2 = np.swapaxes(temp1, 0, 2) # (3, 73257, 32, 32)
	fixed = np.swapaxes(temp2, 0, 1)
	return fixed


