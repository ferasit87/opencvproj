import fuzzy.storage.fcl.Reader
system = fuzzy.storage.fcl.Reader.Reader().load_from_file("demo.fcl")
 
# preallocate input and output values
my_input = {
        "Our_Health" : 0.0,
        "Enemy_Health" : 0.0
        }
my_output = {
        "Aggressiveness" : 0.0
        }
 
# if you need only one calculation you do not need the while
while 1:
        # set input values
        my_input["Our_Health"] = 60.0
        my_input["Enemy_Health"] = 60.0
 
        # calculate
        system.calculate(my_input, my_output)
 
        # now use outputs
        print my_output["Aggressiveness"]