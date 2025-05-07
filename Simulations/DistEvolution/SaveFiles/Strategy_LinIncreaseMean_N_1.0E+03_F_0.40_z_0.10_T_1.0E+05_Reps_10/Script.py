from Params import *

import sys
sys.path.insert(0,'../../CoreFunctions')

import Core


import matplotlib.pyplot as plt

import numpy as np

import copy

import time

"""
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
"""
##########################
###Main Process###########
##########################
Network = Core.Init(N,z,F)

Stats = []
HistEvolutionList = []

for r in range(Repeats):
    Stat,HistEvolution,bin_edges = Core.Evolve(N,z,F,T,Strategy,DistEvolution=True)
    Stats.append(Stat)
    HistEvolutionList.append(HistEvolution)


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

#plt.plot(x,MeanStats[:,2])#min of nonzealots
#plt.plot(x,MeanStats[:,3])#Zealots

print(MeanStats[:,2])

plt.plot(x,np.ones(len(x))*theory_mean,linestyle='dashed')

plt.savefig(SaveDirName+"/F_%0.2f.png"%(F))
plt.close()





#Hists
"""
print(HistEvolutionList)
print(HistEvolutionList[0])


print(len(HistEvolutionList))
print(len(HistEvolutionList[0]))
"""
t = np.arange(0,T,T/10)
for i in range(len(HistEvolutionList[0])):          #for each time step
    HistSum = copy.copy(HistEvolutionList[0][i])    #Baseline
    for j in range(1,Repeats):       #loop over repeats
        HistSum += HistEvolutionList[j][i]
    
    HistMean = HistSum/len(HistEvolutionList)       #Average over the number or repeats

    #print(HistMean)

    plt.figure()
    plt.plot(bin_edges[:-1],HistMean/sum(HistMean))
    plt.savefig(SaveDirName + "/Hist_%d.png"%(int(t[i]/(N*(1-z)))))
    plt.close()







#Saving
Output = SaveDirName + '/datafile.npz'
np.savez(Output,
        F=F,
        N=N,
        z=z,
        T=T,
        Repeats=Repeats,
        Strategy=Strategy,
        Stats = Stats,
        HistEvolutionList=HistEvolutionList,
        bin_edges=bin_edges)

