from Params import *

import sys
sys.path.insert(0,'../../CoreFunctions')

import Core


import matplotlib.pyplot as plt

import numpy as np

import copy

import time


##########################
###Argparser##############
##########################
from argparse import ArgumentParser
parser = ArgumentParser(description='Fitness')
parser.add_argument('-F','--Fitness',type=float,required='True',
        help='Fitness modifying optinions for zealots')
args = parser.parse_args()

F = float(args.Fitness)

FitnessSaveDirName = (SaveDirName + "/F_%0.3f"%(F))

if not os.path.isdir(FitnessSaveDirName):
    os.mkdir(FitnessSaveDirName)
    print("Created Directory for F = ",F)

##########################
###Main Process###########
##########################
Network = Core.Init(N,z,F)

Stats = []

for r in range(Repeats):
    Stats.append(Core.Evolve(N,z,F,T,Strategy))

"""
###Plotting
MeanStats = np.mean(Stats,axis = 0)

x = np.arange(len(Stats[0][:,0]))/(N*(1-z))


theory_mean = 1

if Strategy == "Mean":
    theory_mean = (1-z) / (1 - (1-F)*z)

elif Strategy == "Const":
    theory_mean = F

    if z < (1-F):
        theory_mean = (z*F**2 + 1 - F - z)/((1-z)*(1-F))



plt.figure()
for i in range(Repeats):
    print(F,i)
    plt.plot(x,Stats[i][:,0],alpha = 0.5,color='k')

plt.plot(x,MeanStats[:,0])

plt.plot(x,np.ones(len(x))*theory_mean,linestyle='dashed')

plt.savefig(SaveDirName+"/F_%0.2f.png"%(F))
plt.close()
"""

#Saving
Output = FitnessSaveDirName + '/datafile.npz'
np.savez(Output,
        F=F,
        N=N,
        z=z,
        T=T,
        Repeats=Repeats,
        Strategy=Strategy,
        Stats = Stats)
