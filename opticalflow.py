import numpy as np
import cv2
from threading import Thread
from Queue import Queue
import Queue
import time
import os
import threading
from time import sleep
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import scipy.spatial
import math

def ecldist (p1,p2):
    p = p1-p2
    distances = np.zeros(p.__len__())
    for idx, valx in enumerate(p):
        distances[idx] = np.power(valx, 2)
    result =0
    for idx, valx in enumerate(distances):
        result += valx
    return np.sqrt(result)

def GetAngleDegree (target,origin):
	n= -1
	if (target[1] == origin[1]):
		if (origin[0] > target[0]):
			return 0
		else:
			return math.pi
	n = math.atan2(origin[1] - target[1],origin[0]-target[0])
	print (n)
	return n

def find_neighbors(pindex, triang):
	return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]


def calcLocalscale(valx,neighbors,good_old,good_new):
	Numerator =0
	Denominator = 0
	for idy, valy in enumerate(neighbors):
		Numerator += ecldist(valx,good_new[idy]) - ecldist(valx,good_old[idy])
		Denominator += ecldist(valx,good_old[idy])
	return (Numerator/Denominator)



cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,1024)
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
itr=0
while(1):
    itr+=1
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if itr==70:
        itr=0
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
