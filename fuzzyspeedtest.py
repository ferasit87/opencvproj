import fuzzy.storage.fcl.Reader
import time
from PIL import Image
from PIL import ImageTk
import Tkinter as tki


system = fuzzy.storage.fcl.Reader.Reader().load_from_file("spped.fcl")

# preallocate input and output values
my_input = {
    "Speed": 0.0,
    "Obstacle": 0.0,
    "Target": 0.0
}
my_output = {
    "Acceleration": 0.0
}
my_input["Speed"] = 0.0
my_input["Obstacle"] = 600.0
my_input["Target"] = 600.0

# if you need only one calculation you do not need the while
while 1:
    if (my_input["Target"] < 0.1):
        break;
    time.sleep(0.050)
    # set input values
    my_input["Speed"] = my_input["Speed"] + my_output["Acceleration"]
    # calculate
    system.calculate(my_input, my_output)
    my_input["Target"] = my_input["Target"] - my_input["Speed"]/50
    # now use outputs
    print my_output["Acceleration"]
    print my_input["Speed"]
    print my_input["Target"]