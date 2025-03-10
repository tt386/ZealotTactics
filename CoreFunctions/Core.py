import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def MainProcess(Network,N,z,F,Strategy):

    #Choose index to update opinion
    updating_index = np.random.randint(0,N*(1-z)+1)

    if Strategy == "Targetted":
        Network[-int(N*z):] = F * Network[updating_index]

    # Choose an index with probability proportional to the values
    selected_index = np.random.choice(len(Network), p=Network / Network.sum())

    #Set the new value:
    Network[updating_index] = Network[selected_index]

    #Update Zealots
    if Strategy == "Mean":
        mean_nonzealots = np.mean(Network[:int(N*(1-z))])
        Network[-int(N*z):] = F* mean_nonzealots #*min(Network[:int(N*(1-z))])

    if Strategy =='Max':
        Network[-int(N*z):] = F*max(Network[:int(N*(1-z))])

    if Strategy == 'Min':
        Network[-int(N*z):] = F*min(Network[:int(N*(1-z))])


    return Network


def GetStats(Network,N,z,F):

    mean_nonzealots = np.mean(Network[:int(N*(1-z))])
    max_nonzealots = max(Network[:int(N*(1-z))])
    min_nonzealots = min(Network[:int(N*(1-z))])
    zealot_val = Network[-1]

    return [mean_nonzealots,max_nonzealots,min_nonzealots,zealot_val]



def Init(N,z,F):
    #Network
    Network = np.ones(N)

    #Set zealots
    Network[-int(N*z):] = F

    return Network


def Evolve(N,z,F,T,Strategy):

    Stats = []

    Network = Init(N,z,F)

    checktimes = np.arange(0,T,T/10)
    for t in np.arange(T):
        
        if t in checktimes:
            print(t)
    
        Network = MainProcess(Network,N,z,F,Strategy)

        Stats.append(GetStats(Network,N,z,F))

    return np.asarray(Stats)



def PlotStats(Stats,N,z,F,Strategy):

    Mean = Stats[:,0]
    Max = Stats[:,1]
    Min = Stats[:,2]
    Zealots = Stats[:,3]

    x = np.arange(len(Mean))/(N*(1-z))

    theory_mean = 1

    if Strategy == "Mean":
        theory_mean = (1-z) / (1 - (1-F)*z)

    elif Strategy == "Const":
        theory_mean = F

        if z < (1-F):
            theory_mean = (z*F**2 + 1 - F - z)/((1-z)*(1-F))

    print(theory_mean)
    print(Mean)
    plt.figure()
    plt.plot(x,Mean,label='Mean')
    plt.plot(x,Max,label='Max')
    plt.plot(x,Min,label='Min')
    #plt.plot(x,Zealots,label='Zealot')
    plt.plot(x,np.ones(len(x))*theory_mean,label='Theory Mean')
    plt.title(str(Strategy) + " F=%0.3f, z = %0.3f"%(F,z))
    plt.legend()
    plt.yscale("log")
    plt.show()






def VaryF(N,z,T,Strategy):
    Flist = np.linspace(0.8,1,10)


    MeanList = []

    for F in Flist:
        Stats = Evolve(N,z,F,T,Strategy)
        MeanList.append(Stats[:,0])
        #PlotStats(Stats,N,z,F,Strategy)


    x = np.arange(len(MeanList[0]))/(N*(1-z))

    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=min(Flist),vmax=max(Flist))
    
    fig,ax = plt.subplots()
    for i in range(len(Flist)):
        F = Flist[i]
        plt.plot(x,MeanList[i],color=cmap(norm(F)))#,label=f'F={F:.2f}')

    #Colorbar
    sm = cm.ScalarMappable(cmap=cmap,norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm,ax=ax,ticks=Flist)
    cbar.set_label("F Value")
    cbar.set_ticks(Flist)
    cbar.set_ticklabels([f"{F:.2f}" for F in Flist])

    ax.set_xlabel("Time")
    ax.set_ylabel("Mean Opinion")
    plt.show()
    

"""
N = 1000
T = int(1e5)
F = 0.9
z = 0.1
Strategy="Mean"
"""
"""
Strategies:
    Mean: Zealots adopt the mean of their neighbours * F
    Max: Zealots adopt the max of the neighbours * F
    Min: Zealots adopt the min of the neighbours * F
    Const: Zealots remain at F
    Targetting: Zealots express opinions uniquely to each people: F*their opinion.
"""
"""
Target = 0.5#1-z
if Strategy == "Mean":
    F= (1-z)*(1-Target)/(Target*z)
if Strategy == "Const":
    F = (1-z)*(1-Target)/z
print(F)
"""

"""
Stats = Evolve(N,z,F,T,Strategy)
PlotStats(Stats,N,z,F,Strategy)
"""
"""
VaryF(N,z,T,Strategy)
"""







