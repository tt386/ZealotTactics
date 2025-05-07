import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import sys
import copy

#############
##Argparse###
#############
import os.path

def is_valid_file(parser,arg):
    if not os.path.exists(arg):
        parser.error("The directory %s does not exist"%(arg))
    else:
        return open(arg,'r')

from argparse import ArgumentParser

parser = ArgumentParser(description='Plotting')
parser.add_argument("-d",'--directory',help='The directory of the data')
args = parser.parse_args()

################
#Extract Data###
################
filename = 'datafile.npz'

"""
#Find list of all the datafiles
tempdirlist = os.listdir(args.directory)
dirlist = []
for i in tempdirlist:
    if os.path.isdir(os.path.join(args.directory,i)):
        dirlist.append(os.path.join(args.directory,i))

print("Dirlist:",dirlist)
"""




Flist = []
MeanList = []


try:
    with np.load(os.path.join(args.directory,filename)) as data:
        F = data["F"]
        z = data["z"]
        N = data["N"]
        Stats = data["Stats"]
        Strategy = data["Strategy"]
        T = data["T"]
        bin_edges=data["bin_edges"]
        HistEvolutionList = data["HistEvolutionList"]

except Exception as e: print(e)

#Plotting
MeanStats = np.mean(Stats,axis = 0)
MeanList.append(MeanStats[:,0])

x = np.arange(len(Stats[0][:,0]))/(N*(1-z))


theory_mean = 1

if Strategy == "Mean":
    theory_mean = (1-z) / (1 - (1-F)*z)


elif Strategy == "Const":
    theory_mean = F

    if z < (1-F):
        theory_mean = (z*F**2 + 1 - F - z)/((1-z)*(1-F))


plt.figure()
for j in range(len(Stats)):
    print(F,j)
    plt.plot(x,Stats[j][:,0],alpha = 0.5,color='k')

plt.plot(x,MeanStats[:,0])
plt.plot(x,MeanStats[:,1]) # Max
plt.plot(x,MeanStats[:,2]) # Min
#plt.plot(x,MeanStats[:,3])#Zealots
plt.plot(x,np.ones(len(x))*(1-z))
if Strategy == "Mean":
    #Theory plot
    alpha = (-2*F*z + F + z -1) / (1+(F-1)*z)
    d = F*z*(z-1) / ((F-1)*z + 1)
    p = d * (1- np.exp(alpha * x)) / alpha
    Mean = (1-p-z) /(1-z-F*p)

    plt.plot(x,Mean,alpha=0.7,linestyle='dashed')


plt.plot(x,np.ones(len(x))*theory_mean,linestyle='dashed')

#plt.ylim(1-z,1)
#plt.yscale("log")
plt.savefig(str(args.directory)+"/F_%0.2f.png"%(F))
plt.close()







t = np.arange(0,T,T/10)
for i in range(len(HistEvolutionList[0])):          #for each time step
    HistSum = copy.copy(HistEvolutionList[0][i])    #Baseline
    for j in range(1,len(HistEvolutionList)):       #loop over repeats
        HistSum += HistEvolutionList[j][i]

    HistMean = HistSum/len(HistEvolutionList)       #Average over the number or repeats

    print(HistMean)

    plt.figure()
    plt.plot(bin_edges[:-1],HistMean/sum(HistMean))
    plt.yscale("log")
    #plt.xlim(0.8,1.1)
    plt.savefig(str(args.directory) + "/Hist_%d.png"%(int(t[i]/(N*(1-z)))))
    plt.close()

