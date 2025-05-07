import numpy as np
import os
import shutil

#Number of nodes
N = 1000

#Zealot proportion
z = 0.1


#Fitness
F=0.95


#Time for a simulation 
T = int(1e5)

#Repeats
Repeats = 10

#Strategy
Strategy = "MeanDynamic"


#####################################
###Savefiles and Saving in general###
#####################################
SaveDirName = ("SaveFiles/"+
        "Strategy_" + Strategy +
        "_N_" + '{:.1E}'.format(N) +
        "_F_%0.2f"%(F) + 
        "_z_%0.2f"%(z) +
        "_T_" + '{:.1E}'.format(T) +
        "_Reps_%d"%(Repeats) 
        )

if not os.path.isdir("SaveFiles"):
    os.mkdir("SaveFiles")


if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Param Directory")


shutil.copyfile("Params.py", SaveDirName+'/LocalParams.py')
shutil.copyfile("Script.py", SaveDirName+"/Script.py")

