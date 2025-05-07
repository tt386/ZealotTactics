import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import optimize

def MainProcess(Network,N,z,F,Strategy):

    #Choose index to update opinion
    updating_index = np.random.randint(0,N*(1-z)+1)

    if Strategy == "Targetted":
        Network[-int(N*z):] = F * Network[updating_index]

    if Strategy == "MeanDynamic":
        
        mean_nonzealots = np.mean(Network[:int(N*(1-z))])
        """
        #WHat fitness is it suggested we're at?
        #Develop new fitness
        F = min((1-z) * (1-mean_nonzealots) / (mean_nonzealots*z) + 0.01,1)
        """
        Network[-int(N*z):] = F* mean_nonzealots



    if Strategy == 'LinIncreaseMean':
        mean_nonzealots = np.mean(Network[:int(N*(1-z))])
        Network[-int(N*z):] = np.linspace(F*mean_nonzealots,1,int(N*z))

    if Strategy == 'SuperExIncreaseMean':
        mean_nonzealots = np.mean(Network[:int(N*(1-z))])

        x = np.linspace(0,1,int(N*z))
        #myarray[-int(N*z):] = F*a*np.exp(np.log(1/(F*a))*x**4)

        Network[-int(N*z):] = mean_nonzealots*F * np.exp(np.log((1-z)/(F*mean_nonzealots))*x**6)


    if Strategy in ["MeanMin","MeanTriangleMin"]:
        mean_nonzealots_1 = np.mean(Network[:int(N*(1-z))])

    if Strategy == "MeanTriangle":
        trianglehalfwidth = 0.1
        mean_nonzealots = np.mean(Network[:int(N*(1-z))])

        def triangular_distribution(n,F,w):
            #Uniform spaced values in (0,1) for deterministic approach
            u = np.linspace(0,1,n)

            #Inverse CDF of the triangular dust
            x = np.where(u < 0.5, F-w + np.sqrt(2 * w**2 * u), F + w - np.sqrt(2*w**2*(1-u)))

            x[x<0] = 0
            x[x>1] = 1

            return x

        Network[-int(N*z):] = triangular_distribution(int(z*N),F*mean_nonzealots,trianglehalfwidth)

    # Choose an index with probability proportional to the values
    selected_index = np.random.choice(len(Network), p=Network / Network.sum())

    #Set the new value:
    Network[updating_index] = Network[selected_index]

    #Update Zealots
    if Strategy == "Mean":
        mean_nonzealots = np.mean(Network[:int(N*(1-z))])
        Network[-int(N*z):] = F* mean_nonzealots #*min(Network[:int(N*(1-z))])

    if Strategy == "Median":
        median_nonzealots = np.median(Network[:int(N*(1-z))])

    if Strategy =='Max':
        Network[-int(N*z):] = F*max(Network[:int(N*(1-z))])

    if Strategy == 'Min':
        Network[-int(N*z):] = F*min(Network[:int(N*(1-z))])


    if Strategy == "MeanMin":
        mean_nonzealots_2 = np.mean(Network[:int(N*(1-z))])

        if mean_nonzealots_2 < mean_nonzealots_1:
            Network[-int(N*z):] = F* mean_nonzealots_2
        

    if Strategy == "MeanTriangleMin":
        trianglehalfwidth = 0.05
        mean_nonzealots = np.mean(Network[:int(N*(1-z))])

        def triangular_distribution(n,F,w):
            #Uniform spaced values in (0,1) for deterministic approach
            u = np.linspace(0,1,n)

            #Inverse CDF of the triangular dust
            x = np.where(u < 0.5, F-w + np.sqrt(2 * w**2 * u), F + w - np.sqrt(2*w**2*(1-u)))

            x[x<0] = 0
            x[x>1] = 1

            return x

        Network[-int(N*z):] = triangular_distribution(int(z*N),F*min(mean_nonzealots_1,mean_nonzealots),trianglehalfwidth)


    return Network


def GetStats(Network,N,z,F):

    mean_nonzealots = np.mean(Network[:int(N*(1-z))])
    max_nonzealots = max(Network[:int(N*(1-z))])
    min_nonzealots = min(Network[:int(N*(1-z))])
    zealot_val = Network[-1]

    return [mean_nonzealots,max_nonzealots,min_nonzealots,zealot_val]



def Init(N,z,F,Strategy = "Const",trianglehalfwidth = 0.1,InitDict={}):
    #Network
    Network = np.ones(N)

    #Set zealots
    Network[-int(N*z):] = F


    if Strategy == "ConstTriangle":
        def triangular_distribution(n,F,w):
            #Uniform spaced values in (0,1) for deterministic approach
            u = np.linspace(0,1,n)

            #Inverse CDF of the triangular dust
            x = np.where(u < 0.5, F-w + np.sqrt(2 * w**2 * u), F + w - np.sqrt(2*w**2*(1-u)))

            x[x<0] = 0
            x[x>1] = 1

            return x

        Network[-int(N*z):] = triangular_distribution(int(z*N),F,trianglehalfwidth)


    if Strategy == "BiModal":
        z1 = InitDict["z1"]
        F1 = InitDict["F1"]
        F2 = InitDict["F2"]
        p1 = InitDict["p1"]
        p2 = InitDict["p2"]


        Network[-int(N*z):] = F1
        Network[-int(N*(z-z1)):] = F2

        Network[:int(N*(p1+p2))] = F2
        Network[:int(N*(p1))] = F1


    return Network


def Evolve(N,z,F,T,Strategy,Network = "FALSE",CheckTimeNum = 10,DistEvolution=False):

    Stats = []

    if Network == "FALSE":
        Network = Init(N,z,F)

    checktimes = np.arange(0,T,T/CheckTimeNum)

    if DistEvolution:
        # Define bin edges based on min/max opinions over all time steps
        num_bins = 1000
        opinion_min, opinion_max = 0, 1.1
        bin_edges = np.linspace(opinion_min, opinion_max, num_bins + 1)

        distribution_over_time = np.zeros((len(checktimes), num_bins))


    #Main Process
    checktimecounter = 0
    for t in np.arange(T):
        
        if t in checktimes:
            print(t)
            if DistEvolution:
                #counts, _ = np.histogram(Network, bins=bin_edges)  # Bin the opinions at time t
                counts_nozealots,_ = np.histogram(Network[:int(N*(1-z))],bins=bin_edges)
                distribution_over_time[checktimecounter] = counts_nozealots
                checktimecounter += 1

        if Strategy == "MeanDynamic":
            def a(F):
                return (1+2*F*z-F-z) / (1- (1-F)*z)


            def b(F):
                return (F*z*(1-z)) / (1-(1-F)*z)


            def M(F,t):
                return (b(F) * (1-np.exp(-a(F)*t)) + a(F)*(z-1)) / (F*b(F) * (1-np.exp(-a(F)*t)) + a(F)*(z-1))

            F = optimize.minimize(M,x0=0.5,args=(t/(N*(1-z))),bounds=[(0,1)]).x[0]

        Network = MainProcess(Network,N,z,F,Strategy)

        Stats.append(GetStats(Network,N,z,F))

    if DistEvolution:
        return (np.asarray(Stats),distribution_over_time,bin_edges)

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
    MeanDynamic: Same as above, but F increases when the expcted mean is achieved
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







