# import the necessary packages
from __future__ import print_function
from PIL import Image
from PIL import ImageTk
import Tkinter as tki
import threading
import datetime
import imutils
import cv2
import os
import numpy as np
import math
import scipy.spatial



def ecldist(p1, p2):
    p = p1 - p2
    distances = np.zeros(p.__len__())
    for idx, valx in enumerate(p):
        distances[idx] = np.power(valx, 2)
    result = 0
    for idx, valx in enumerate(distances):
        result += valx
    return np.sqrt(result)


def GetAngleDegree(target, origin):
    n = -1
    if (target[1] == origin[1]):
        if (origin[0] > target[0]):
            return 0
        else:
            return math.pi
    else:
        n = math.atan2(origin[1] - target[1], origin[0] - target[0])
    print(n)
    return n

def calcLocalscale(valx, neighbors, good_old, good_new):
    Numerator = 0
    Denominator = 0
    newNeighbors = neighbors
    for idy, valy in enumerate(newNeighbors):
        Numerator += ecldist(valx, good_new[valy]) - ecldist(valx, good_old[valy])
        Denominator += ecldist(valx, good_old[valy])
    if (Denominator == 0):
        return 0 ;
    return (Numerator / Denominator)

def find_neighbors(pindex, triang):
    return triang.neighbors[1][triang.neighbors[pindex][0]:triang.neighbors[pindex][1] : triang.neighbors[pindex][2]]


class PhotoBoothApp:
    def __init__(self, vs, outputPath):
        # store the video stream object and output path, then initialize
        # the most recently read frame, thread for reading frames, and
        # the thread stop event
        self.vs = vs
        self.outputPath = outputPath
        self.frame = None
        self.thread = None
        self.stopEvent = None

        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None
        # create a button, that when pressed, will take the current
        # frame and save it to file
        btn = tki.Button(self.root, text="Snapshot!",
                         command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10,
                 pady=10)

        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.videoLoop, args=())
        self.thread.start()

        # set a callback to handle when the window is closed
        self.root.wm_title("PyImageSearch PhotoBooth")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def videoLoop(self):
        # DISCLAIMER:
        # I'm not a GUI developer, nor do I even pretend to be. This
        # try/except statement is a pretty ugly hack to get around
        # a RunTime error that Tkinter throws due to threading
        cap = self.vs
        # cap.stream.set(3, 1280)
        # cap.set(4, 1024)
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.1,
                              minDistance=17,
                              blockSize=7)

        # Parameters for lucas kanade optical flow
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Create some random colors
        color = np.random.randint(0, 255, (100, 3))

        # Take first frame and find corners in it
        old_framearr = cap.read()
        old_frame = old_framearr[1]
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        locals = np.zeros(np.size(p0, 0))
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        itr = 0
        try:
            # keep looping over frames until we are instructed to stop
            while not self.stopEvent.is_set():
                # grab the frame from the video stream and resize it to
                # have a maximum width of 300 pixels
                framearr = cap.read()
                frame = framearr[1]
                # self.frame = imutils.resize(self.frame, width=600)

                # OpenCV represents images in BGR order; however PIL
                # represents images in RGB order, so we need to swap
                # the channels, then convert to PIL and ImageTk format
                ## image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


                # if the panel is not None, we need to initialize it

                ret = frame
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

                # Select good points
                try:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                except BaseException, e:
                    print("[INFO] caught a RuntimeError")
                # draw the tracks
                #if (itr == 1):
                    #tri = scipy.spatial.Delaunay(good_old)
                    #for idx, valx in enumerate(good_new):
                        #neighbors = find_neighbors(idx, tri)
                        #locals[idx] = calcLocalscale(p0[idx][0], neighbors, good_old, good_new)

                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                img = cv2.add(frame, mask)

                # cv2.imshow('frame', img)
                image = Image.fromarray(img)
                image = ImageTk.PhotoImage(image)
                self.frame = img.copy()
                # Now update the previous frame and previous points
                if self.panel is None:
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)

                    # otherwise, simply update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
                itr = itr + 1
                if itr == 170:
                    itr = 0
                    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        except RuntimeError, e:
            print("[INFO] caught a RuntimeError")

    def takeSnapshot(self):
        # grab the current timestamp and use it to construct the
        # output path
        ts = datetime.datetime.now()
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((self.outputPath, filename))

        # save the file
        cv2.imwrite(p, self.frame.copy())
        print("[INFO] saved {}".format(filename))

    def onClose(self):
        # set the stop event, cleanup the camera, and allow the rest of
        # the quit process to continue
        print("[INFO] closing...")
        self.stopEvent.set()
        print("[INFO] closing...")
        # self.vs.stop()

        print("[INFO] closing...")
        self.root.quit()
        print("[INFO] closed...")
