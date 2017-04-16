import sys,os

try:
    import StringIO
except:
    import io as StringIO
	
import fuzzy.OutputVariable
import fuzzy.InputVariable
import fuzzy.defuzzify.Dict
import fuzzy.fuzzify.Dict
import fuzzy.set.Function
import fuzzy.storage.fcl.Reader

system = fuzzy.storage.fcl.Reader.Reader().load_from_file("demo.fcl")

my_input = {
        "TimeDay": 12,
        "ApplicateHost": 20,
        "TimeConfiguration": 5,
        "TimeRequirements": 5
        }
my_output = {
        "ProbabilityDistribution": 0.2,
        "ProbabilityAccess": 0.2
        }
        
system.calculate(my_input, my_output)

print(u"ProbabilityAccess = {0}".format(my_output["ProbabilityAccess"]))