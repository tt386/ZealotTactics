import numpy as np
import os
import shutil

#Number of nodes
N = 1000

#Zealot proportion
z = 0.1

#Time for a simulation 
T = int(1e5)

#Repeats
Repeats = 100

#Strategy
Strategy = "Max"

#F values
Flist = np.linspace(0.1,1,21)

#####################################
###Savefiles and Saving in general###
#####################################
SaveDirName = ("SaveFiles/"+
        "Strategy_" + Strategy +
        "_N_" + '{:.1E}'.format(N) +
        "_z_%0.2f"%(z) +
        "_T_" + '{:.1E}'.format(T) +
        "_Reps_%d"%(Repeats) + 
        "_MinF_%0.2f_MaxF_%0.2f_FNum_%d"%(min(Flist),max(Flist),len(Flist))
        )

if not os.path.isdir("SaveFiles"):
    os.mkdir("SaveFiles")


if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Param Directory")


shutil.copyfile("Params.py", SaveDirName+'/LocalParams.py')
shutil.copyfile("Script.py", SaveDirName+"/Script.py")

