# import the necessary packages
from __future__ import print_function
import Tkinter as tki
import threading
import time
import fuzzy.storage.fcl.Reader




def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
tki.Canvas.create_circle = _create_circle

class CanvasThread:
    def __init__(self):
        self.my_input = {
            "Speed": 0.0,
            "Obstacle": 0.0,
            "Target": 0.0
        }
        self.my_output = {
            "Acceleration": 0.0
        }
        self.my_input["Speed"] = 0.0
        self.my_input["Obstacle"] = 600.0
        self.my_input["Target"] = 900.0
        self.system = fuzzy.storage.fcl.Reader.Reader().load_from_file("spped.fcl")
        self.thread = None
        self.stopEvent = None
        # initialize the root window and image panel
        self.root = tki.Tk()
        self.panel = None
        self.canvas = tki.Canvas(self.root)
        self.canvas.pack(fill=tki.BOTH, expand=1)  # Stretch canvas to root window size.
        # start a thread that constantly pools the video sensor for
        # the most recently read frame
        self.canvas.configure(bg="#cee5ed")
        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.drawCanvas, args=())
        self.thread.start()
        self.root.wm_geometry("1200x750")
        # set a callback to handle when the window is closed
        self.root.wm_title("Draw canvas")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    def drawCanvas(self):
        # background_image=Tk.PhotoImage(file="test.png")
        # image = canvas.create_image(0, 0, anchor=Tk.NW, image=background_image)
        i = 0
        line = self.canvas.create_line(10, 300, 1150, 300, fill="#99e6bb" , width= 150)
        line = self.canvas.create_line(910, 225, 910, 375, fill="green", width=2)
        cercle = self.canvas.create_circle(10, 300, 5, fill="red", outline="")
        while not self.stopEvent.is_set():
            print(self.my_output["Acceleration"])
            print(self.my_input["Speed"])
            print(self.my_input["Target"])
            if (self.my_input["Target"] < 3):
                break;
            time.sleep(0.01)
            self.my_input["Speed"] = self.my_input["Speed"] + self.my_output["Acceleration"]
            # calculate
            self.system.calculate(self.my_input, self.my_output)
            self.my_input["Target"] = self.my_input["Target"] - self.my_input["Speed"] / 50
            self.canvas.move(cercle , self.my_input["Speed"]/50, 0)
    def takeSnapshot(self):
            print("[INFO] saved {}")
    def onClose(self):
            # set the stop event, cleanup the camera, and allow the rest of
            # the quit process to continue
            print("[INFO] closing...")
            self.stopEvent.set()
            print (self.stopEvent.isSet())
            print("[INFO] closing...")
            self.root.quit()
            print("[INFO] closed...")
