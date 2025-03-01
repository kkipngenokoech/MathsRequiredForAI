import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection, plotMatches
#Complete functions above this line before this step
def matchPics(I1, I2):
	'''
	Inputs:
	I1, I2 : Images to match
	
	Outputs:
	matches : Array of matched features
	locs1 : Locations of matched features in Image 1
	locs2 : Locations of matched features in Image 2
 
	Description:
	This function takes as input two images and returns the matches between them.
 
	'''
	
	#I1, I2 : Images to match

	#Convert Images to GrayScale
	I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
	I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	
	#Detect Features in Both Images
	locs1 = corner_detection(I1_gray)
	locs2 = corner_detection(I2_gray)
	
	
	#Obtain descriptors for the computed feature locations
	desc1, locs1 = computeBrief(I1_gray, locs1)
	desc2, locs2 = computeBrief(I2_gray, locs2)
	

	#Match features using the descriptors
	matches = briefMatch(desc1, desc2)
	

	return matches, locs1, locs2

I1 = cv2.imread('../data/cv_cover.jpg')
I2 = cv2.imread('../data/cv_desk.png')

matches, locs1, locs2 = matchPics(I1, I2)
plotMatches(I1, I2, matches, locs1, locs2)