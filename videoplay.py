import numpy as np
import cv2
from Tkinter import *


root = Tk()
topFrame = Frame(root)
topFrame.pack()
bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM)


botton1 = Button(topFrame, text="botton1", fg='red')
botton2 = Button(topFrame, text="botton2", fg='green')
botton3 = Button(topFrame, text="botton3", fg='blue')
botton4 = Button(bottomFrame, text="botton4", fg='purple')

botton1.pack()
botton2.pack()
botton3.pack()
botton4.pack()

root.mainloop()

import  PIL
import tkFileDialog

#Tkinter._test()
#print Tkinter.TclVersion
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()