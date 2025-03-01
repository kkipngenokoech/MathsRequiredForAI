import numpy as np
import cv2
from matchPics import matchPics


#Q3.5
#Read the image and convert to grayscale, if necessary, you can use OpenCV
cover = cv2.imread('data/cv_cover.jpg')
cover = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)

for i in range(36):
	#Rotate Image
	rotated = cv2.warpAffine(cover, cv2.getRotationMatrix2D((cover.shape[1]/2, cover.shape[0]/2), i*10, 1), (cover.shape[1], cover.shape[0]))
	
	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cover, rotated)

	#Update histogram
	if i == 0:
		hist, bins = np.histogram(matches, bins=10, range=(0, 10))
	else:
		hist = np.add(hist, np.histogram(matches, bins=10, range=(0, 10))[0])


#Display histogram
import matplotlib.pyplot as plt
plt.bar(range(10), hist)
plt.xlabel('Number of Matches')
plt.ylabel('Number of Images')
plt.title('Histogram of Matches')
plt.show()

