from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
from threading import Thread
from Queue import Queue
import Queue
import time
import RPi.GPIO as GPIO
import numpy as np
import serial
import os
import threading
from time import sleep
import scipy.spatial
import math




def thread1(threadname, q):
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
#n = n % math.pi
print (n)
return n
#read variable "a" modify by thread 2
targetPoint = np.zeros(2)
targetPoint[0] = 400
targetPoint[1] = 200
CurrentPoit = np.zeros(2)
CurrentPoit[0] = 400
CurrentPoit[1] = 200
distance = 20
alpha = 0.0
GPIO.setmode(GPIO.BOARD)
GPIO.setup(33, GPIO.IN)
GPIO.setup(35, GPIO.IN)
GPIO.setup(40, GPIO.IN)
GPIO.setup(38, GPIO.IN)
OldsignL = 0
OldsignR = 0
totalroundL = 0
totalroundR = 0
oldticks = time.time()
oldnumRoundPSL = 0
oldnumRoundPSR = 0
time.sleep(1)
ser = serial.Serial(
port='/dev/ttyUSB0',
baudrate = 9600,
parity=serial.PARITY_NONE,
stopbits=serial.STOPBITS_ONE,
bytesize=serial.EIGHTBITS,
timeout=1
)
while (True):
spedL = 0
spedR = 0
angel = 0
bytesToRead = ser.inWaiting()
input = ser.readline()
arrinput = input.split(',')
xx= 0
yy = 0
#run= 0
if (len(input) > 1):
print('reading ...')
print('input len'+str(len(input)))
if (len(arrinput) == 4):
if (arrinput[0] == 'x'):
targetPoint[0] = int(arrinput[1])
xx = 1
if (arrinput[2] == 'y'):
targetPoint[1] = int(arrinput[3])
yy = 1
if ((xx==1) &(yy == 1) | (distance > 15) ):
print('get the command')
print('CurrentPoit=============================:'+str(CurrentPoit[0])+str(CurrentPoit[1]))
print('targetPoint============================= :'+str(targetPoint[0])+str(targetPoint[1]))
distance = ecldist(CurrentPoit, targetPoint)
print('diatance = '+str( distance))
if (distance > 10):
angel = GetAngleDegree(CurrentPoit, targetPoint)
print ('angelcos' + str (math.cos(math.fabs(alpha - angel))))
if ((math.cos(math.fabs(alpha - angel)) < 0.99)):
rotateSpeed = math.fabs(alpha-angel) * 5.8 / math.pi
if (rotateSpeed < 0.15):
rotateSpeed = 0.15
if (((angel > -math.pi) & (angel <= 0) & (angel < alpha)) | ((angel <= math.pi) & (angel >= 0) & (angel < alpha))):
speed = rotateSpeed
move = 'l'
else:
move = 'r'
speed = rotateSpeed
else:
speed = float(distance/220)
if (speed < 0.15):
speed = 0.15
move = 'f'

else :
move = -1
speed = 0
print('speed in encoder '+str(speed))
print('move in encoder '+str(move))
print ('alpha='+str(alpha))
dc = 0.0
q.put(move)
q.put(speed)
q.put(speed)
run= 1
while (run <> 2):
first = GPIO.input(33)
second = GPIO.input(35)
first2 = GPIO.input(38)
second2 = GPIO.input(40)
NewsignL = first2* 2 + second2
NewsignR = first* 2 + second
x = np.array( [0,-1,1,2,1,0,2,-1,-1,2,0,1,2,1,-1,0], np.int32)
OutL = x [OldsignL * 4 + NewsignL]
OutR = x [OldsignR * 4 + NewsignR]
totalroundL += OutL
totalroundR += OutR
CurrentnumRoundPSL = totalroundL - oldnumRoundPSL
CurrentnumRoundPSR = totalroundR - oldnumRoundPSR
currentticks = time.time()
Tic = currentticks - oldticks
if (Tic > 0.1):
speedL =
(CurrentnumRoundPSL)/(Tic*20)
if (math.fabs(speedL)>1):
speedL +=100*speedL/math.fabs(speedL)
speedL /=10
speedL +=100*speedL/math.fabs(speedL)
speedL /=10
newspeed = float (int(speedL))
speedL = newspeed + (speedL/100)
else:
speedL =0.0
speedR = (CurrentnumRoundPSR)/(Tic*20)
if (math.fabs(speedR)>1):
speedR += 100*speedR/math.fabs(speedR)
speedR /=10
speedR +=100*speedR/math.fabs(speedR)
speedR /=10
newspeed = float (int(speedR))
speedR = newspeed + (speedR/100)
else:
speedR =0.0

print('L,'+str(speedL)+',R,'+str(speedR)+'\n')
print(speedL - speedR)
dc = (speedL + speedR) / 2
cx = int(CurrentPoit[0] + (dc * math.cos(alpha)))
cy = int(CurrentPoit[1] + (dc * math.sin(alpha)))
alpha += ((speedL - speedR)/440);
if (alpha > math.pi):
alpha = -1* alpha
if (math.fabs(alpha) > (2*math.pi - 0.1)):
alpha = alpha % (2 * math.pi)
CurrentPoit[0] = cx
CurrentPoit[1] = cy
print ('alpha='+str(alpha))
CurrentnumRoundPSL= totalroundL
CurrentnumRoundPSR= totalroundR
oldnumRoundPSL = totalroundL
oldnumRoundPSR = totalroundR
oldticks = currentticks
OldsignL = NewsignL
OldsignR = NewsignR

try :
size = q.qsize()
if (size == 1):
run = q.get()
#print(run)
except Exception:
i=0
if (run==2):
print('break')

def thread2(threadname, q):
feature_params = dict( maxCorners = 100,
qualityLevel = 0.3,
minDistance = 7,
blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15,15),
maxLevel = 2,
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
def find_neighbors(pindex, triang):
return triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]
def ecldist (p1,p2):
p = p1-p2
distances = np.zeros(p.__len__())
for idx, valx in enumerate(p):
distances[idx] = np.power(valx, 2)
result =0
for idx, valx in enumerate(distances):
result += valx
return np.sqrt(result)

def calcLocalscale(valx,neighbors,good_old,good_new):
Numerator =0
Denominator = 0
for idy, valy in enumerate(neighbors):
Numerator += ecldist(valx,good_new[idy]) - ecldist(valx,good_old[idy])
Denominator += ecldist(valx,good_old[idy])

#print(Numerator)
#print(Denominator)
return (Numerator/Denominator)

def tuningGoodFeaturesToTrack (p0):
return p0

def goaheed():
rawCapture = PiRGBArray(camera, size=(640, 480))
camera.capture_continuous(rawCapture, format="bgr", use_video_port=True)
# capture frames from the camera
itr = 0
time.sleep(0.5)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
if (itr <10):
if (itr < 4):
forward(0.3)
if (itr>18):
reverse(0.3)
#print('has moved')
oldticks = time.time()
itr += 1
image = cv2.flip(frame.array,0)
#print (str(itr))
#rawCapture.truncate(0)
if (itr == 1):
old_frame = image
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
oldimgcrop = old_gray[200:460, 100:500]
p0 = cv2.goodFeaturesToTrack(oldimgcrop, mask = None, **feature_params)
for idx, valx in enumerate(p0):
p0[idx][0][0] = p0[idx][0][0] + 100
p0[idx][0][1] = p0[idx][0][1] + 200
#p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
p0 = tuningGoodFeaturesToTrack(p0)
locals = np.zeros(np.size(p0,0))
x = np.array( [[[130 ,100]],[[120, 200]],[[430 , 230]],[[420 , 230]],[[410 , 230]],[[400 , 230]],[[430 , 220]]], np.float32)
# Create a mask image for drawing
purposes
mask = np.zeros_like(old_frame)
frame = image
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# calculate optical flow
p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
# Select good points
good_new = p1[st==1]
good_old = p0[st==1]
if (itr == 1):
tri = scipy.spatial.Delaunay(good_old)
# draw the tracks

for idx, valx in enumerate(good_new):
neighbors = find_neighbors(idx,tri)
locals[idx] = calcLocalscale(p0[idx][0],neighbors,good_old,good_new)
#print(locals)
for i,(new,old) in enumerate(zip(good_new,good_old)):
a,b = new.ravel()
c,d = old.ravel()
mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
img = cv2.add(frame,mask)
old_gray = frame_gray.copy()
p0 = good_new.reshape(-1,1,2)
newticks = time.time()
rawCapture.seek(0)
rawCapture.truncate()
#print ('time ='+str(newticks-oldticks))
else:
break

cv2.imwrite("face-.jpg", img)
print('writen !! ')
def rotateRight (x):
GPIO.output(11, GPIO.HIGH)
GPIO.output(15, GPIO.HIGH)
sleep(x)
GPIO.output(11, GPIO.LOW)
GPIO.output(15, GPIO.LOW)
def rotateLeft (x):
GPIO.output(7, GPIO.HIGH)
GPIO.output(13, GPIO.HIGH)
sleep(x)
GPIO.output(7, GPIO.LOW)
GPIO.output(13, GPIO.LOW)
def forward (x):
GPIO.output(7, GPIO.HIGH)
GPIO.output(15, GPIO.HIGH)
sleep(x)
GPIO.output(7, GPIO.LOW)
GPIO.output(15, GPIO.LOW)
def GetAngleDegree (p1,p2):
if (p1[1] == p2[1]):
if (p2[0] > p1[0]):
return math.pi
else:
return 0
n = math.atan2(p2[1] - p1[1],p2[2]-p1[0])
n = n % math.pi
return n

#color = np.random.randint(0,255,(100,3))
#camera = PiCamera()
#camera.brightness = 50
#camera.resolution = (640, 480)
#camera.framerate = 5
GPIO.setup(7, GPIO.OUT)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)
GPIO.setup(33, GPIO.IN)
GPIO.setup(35, GPIO.IN)
GPIO.setup(40, GPIO.IN)
GPIO.setup(38, GPIO.IN)
while (True):
size = q.qsize()
if ( size >= 2):
queueLock.acquire()
speed =0
move = 0
run = 1
try :
move= q.get()
print('move in 2 '+str(move))
except Exception:
move=0
try :
speed= q.get ()
print('speed in 2 '+str(speed))
except Exception:
speed=0
queueLock.release()
if (move=='l'):
rotateLeft(speed)
print ('rLeft'+str(speed))
#sleep(speed)
run = 2
elif (move=='r'):
rotateRight(speed)
print ('rRight'+str(speed))
#sleep(speed)
run = 2
elif (move=='f'):
forward(speed)
#sleep(speed)
print ('forward'+str(speed))
run = 2
elif (move==-1):
print ('stoped'+str(move))
run = 2
q.put(run)



queueLock = threading.Lock()
queue = Queue.Queue(2)
thread1 = Thread( target=thread1, args=("Thread-1", queue) )
thread1.start()
thread2 = Thread( target=thread2, args=("Thread-2", queue) )
thread2.start()
thread2.join()
thread1.join()