import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import os
import sys


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

#Find list of all the datafiles
tempdirlist = os.listdir(args.directory)
dirlist = []
for i in tempdirlist:
    if os.path.isdir(os.path.join(args.directory,i)):
        dirlist.append(os.path.join(args.directory,i))

print("Dirlist:",dirlist)


Flist = []
MeanList = []


for i in dirlist:
    try:
        with np.load(os.path.join(i,filename)) as data:
            F = data["F"]
            z = data["z"]
            N = data["N"]
            Stats = data["Stats"]
            Strategy = data["Strategy"]

            Flist.append(F)

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

    if Strategy == "Mean" or "Median":
        #Theory plot
        alpha = (-2*F*z + F + z -1) / (1+(F-1)*z)
        d = F*z*(z-1) / ((F-1)*z + 1)
        p = d * (1- np.exp(alpha * x)) / alpha
        Mean = (1-p-z) /(1-z-F*p)

        plt.plot(x,Mean,alpha=0.7,linestyle='dashed')


    plt.plot(x,np.ones(len(x))*theory_mean,linestyle='dashed')

    plt.savefig(str(i)+"/F_%0.2f.png"%(F))
    plt.savefig(str(args.directory)+"/F_%0.2f.png"%(F))
    plt.close()




Flist,MeanList = zip(*sorted(zip(Flist,MeanList)))

Flist = np.asarray(Flist)
MeanList = np.asarray(MeanList)

plt.figure()
x = np.arange(len(MeanList[0]))/(N*(1-z))

cmap = cm.viridis
norm = mcolors.Normalize(vmin=min(Flist),vmax=max(Flist))

fig,ax = plt.subplots()
for i in range(len(Flist)):
    F = Flist[i]
    plt.plot(x,MeanList[i],color=cmap(norm(F)),alpha = 0.5)#,label=f'F={F:.2f}')


    #Theory plot
    alpha = (-2*F*z + F + z -1) / (1+(F-1)*z)
    d = F*z*(z-1) / ((F-1)*z + 1)
    p = d * (1- np.exp(alpha * x)) / alpha
    Mean = (1-p-z) /(1-z-F*p)

    plt.semilogy(x,Mean,color='k',alpha=0.7,linestyle='dashed')

    #Theory
    if Strategy == "Mean" or Strategy == "Median":
        theory_mean = (1-z) / (1 - (1-F)*z)
        plt.plot(x,np.ones(len(x))*theory_mean,color=cmap(norm(F)),alpha=0.2,linestyle='dashed')

#Colorbar
sm = cm.ScalarMappable(cmap=cmap,norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm,ax=ax,ticks=Flist)
cbar.set_label("F Value")
cbar.set_ticks(Flist)
cbar.set_ticklabels([f"{F:.2f}" for F in Flist])

ax.set_xlabel("Time")
ax.set_ylabel("Mean Opinion")
#plt.xscale("log")
plt.yscale("log")
#plt.ylim(0.98,1)

plt.savefig(str(args.directory) + "/Fig.png")




