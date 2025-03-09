# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 16:02:25 2025

@author: tt386
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import time
from scipy.optimize import curve_fit

np.random.seed(1)

#For Complete

#Number of nodes
N = 100

#Proportion of zealots
z = 0.6

#Zealot strength
F = 0.9

starttime = time.time()



def meanfit(x,b):
    return np.exp(-b*x)

def logmeanfit(x,b):
    return -b * x

def FitSlope(meanlist,N,z,T,SlopeCutoff):
    x = np.arange(len(meanlist))
    
    popt, pcov = curve_fit(logmeanfit, x[:int(T*SlopeCutoff)]/(N*(1-z)), np.log(meanlist[:int(T*SlopeCutoff)]))
    
    print(popt)
    return popt

def MainProcess(N,z,F,T,SlopeCutoff,FlexZealots=True):
    
    #Network
    Network = np.ones(N)
    
    #Set zealots
    Network[-int(N*z):] = F
    
    
    #Main Process
    meanlist = []
    maxlist = []
    minlist = []
    
    
    checktimes = np.arange(0,T,T/10)
    print(checktimes)
    for t in np.arange(T):
        if t in checktimes:
            print(t)
        
        #Choose index to update opinion
        updating_index = np.random.randint(0,N*(1-z)+1)
        
        # Choose an index with probability proportional to the values
        selected_index = np.random.choice(len(Network), p=Network / Network.sum())
    
        #Set the new value:
        Network[updating_index] = Network[selected_index]
        
        #Calculate the new mean
        mean_nonzealots = np.mean(Network[:int(N*(1-z))])
        meanlist.append(mean_nonzealots)
        maxlist.append(max(Network))
        minlist.append(min(Network))
        
        #Update Zealots
        if FlexZealots:
            Network[-int(N*z):] = F* mean_nonzealots #*min(Network[:int(N*(1-z))])
        
        
        
    popt = FitSlope(meanlist,N,z, T, SlopeCutoff)
    
    
    x = np.arange(len(meanlist)) / (N*(1-z))

    theory_mean = (1-z) / (1 - (1-F)*z)

    if not FlexZealots:
        if z < (1-F):
            theory_mean = (z*F**2 + 1 - F - z)/((1-z)*(1-F))
        else:
            theory_mean = F
    
    plt.semilogy(x,meanlist,label='mean')
    plt.plot(x,np.ones(len(x))*theory_mean,label='Theory Mean')
    plt.plot(x,meanfit(x, popt[0]),linestyle='dashed',label='Mean fit')
    plt.plot(x,maxlist,label='max')
    plt.plot(x,minlist,label='min')
    plt.title("F=%0.3f, z = %0.3f"%(F,z))
    plt.legend()
    plt.yscale("log")
    plt.show()

    return meanlist,popt[0],Network


def VaryN(): 
    #########################
    #### N variation#########
    #########################
    #Number of nodes
    N = 100
    
    #Proportion of zealots
    z = 0.6
    
    #Zealot strength
    F = 0.9
    
    #Time 
    T = 10000
    
    NList = np.logspace(2,4,5).astype(int)
    SlopeCutoffList = [1,0.1,1,1,1]
    
    SlopeList = []
    for i in range(len(NList)):
        SlopeList.append(MainProcess(NList[i], z, F,T,SlopeCutoffList[i])[1])
    
    
    def NSlopeFit(x,a,b):
        return a * x**b
    
    popt, pcov = curve_fit(NSlopeFit, NList, SlopeList)
    
    print(popt)
    
    plt.figure()
    plt.loglog(NList,SlopeList)
    plt.loglog(NList,NSlopeFit(NList, popt[0], popt[1]),linestyle='dashed')
    plt.show()

def VaryF():
    #########################
    #### F variation#########
    #########################
    #Number of nodes
    N = 10000
    
    #Proportion of zealots
    z = 0.6
    
    #Zealot strength
    F = 0.9
    
    #Time 
    T = 10000
    
    FList = np.linspace(0.8,0.99,10)
    SlopeCutoffList = np.ones(len(FList))#[1,0.1,1,1,1]
    SlopeCutoffList[0] = 0.2
    SlopeCutoffList[1] = 0.4
    SlopeCutoffList[2] = 0.4
    SlopeCutoffList[4] = 0.6
    
    
    FList = np.linspace(0.1,0.99,10)
    SlopeCutoffList = np.ones(len(FList))#[1,0.1,1,1,1]
    SlopeCutoffList[0] = 0.05
    SlopeCutoffList[1] = 0.05
    SlopeCutoffList[2] = 0.1
    SlopeCutoffList[3] = 0.1
    SlopeCutoffList[4] = 0.1
    SlopeCutoffList[5] = 0.1
    SlopeCutoffList[6] = 0.2
    SlopeCutoffList[7] = 0.2
    
    
    SlopeList = []
    for i in range(len(FList)):
        SlopeList.append(MainProcess(N, z, FList[i],T,SlopeCutoffList[i])[1])
    
    
    
    def func(z,F):
        #return z*(1-F)*F / (N * (1-z) * (1 - (1-F)*z))
        return (1-F)*F*z / ((1+F*z)) 
    
    plt.figure()
    plt.semilogy(FList,SlopeList)
    plt.semilogy(FList,func(z, FList),linestyle='dashed')
    #.loglog(NList,NSlopeFit(NList, popt[0], popt[1]),linestyle='dashed')
    plt.show()

def Varyz():
    #########################
    #### z variation#########
    #########################
    #Number of nodes
    N = 10000
    
    #Proportion of zealots
    z = 0.6
    
    #Zealot strength
    F = 0.9
    
    #Time 
    T = 200000
    
    zList = np.linspace(0.1,0.9,10)
    SlopeCutoffList = np.ones(len(zList))#[1,0.1,1,1,1]
    """
    SlopeCutoffList[-1] = 0.1
    SlopeCutoffList[-2] = 0.3
    SlopeCutoffList[-3] = 0.3
    SlopeCutoffList[-4] = 0.8
    SlopeCutoffList[-6] = 0.8
    SlopeCutoffList[-7] = 0.5
    """
    
    SlopeCutoffList *= 0.5
    print(SlopeCutoffList)
    SlopeList = []
    for i in range(len(zList)):
        SlopeList.append(MainProcess(N, zList[i], F,T,SlopeCutoffList[i])[1])
    
    
    def zSlopeFit(x,a,b):
        return np.exp(b*x)
    
    def logzSlopeFit(x,a,b):
        return a + b*x
    
    popt, pcov = curve_fit(logzSlopeFit, zList, np.log(SlopeList))
    
    print(popt)
    
    def func(z,F):
        #return z*(1-F)*F / (N * (1-z) * (1 - (1-F)*z))
        return (1-F)*F*z / ((1+F*z))
    
    plt.figure()
    plt.semilogy(zList,SlopeList)
    plt.semilogy(zList,func(zList, F),linestyle='dashed')
    #plt.plot(zList,zSlopeFit(zList, np.exp(popt[0]), popt[1]),linestyle='dashed')
    plt.show()



def MeanofMeans():
    #Number of nodes
    N = 1000
    
    #Proportion of zealots
    z = 0.3
    
    #Zealot strength
    F = 0.9
    
    #Time 
    T = 100000
    
    Hist = []
    for Repeats in range(20):
        Hist.append(MainProcess(N, z, F,T,1)[0])
        
    plt.figure()
    
    averaged = np.mean(Hist,axis=0)
    median = np.median(Hist,axis=0)
    
    
    plt.semilogy(np.arange(len(averaged)),averaged,color='k')
    plt.semilogy(np.arange(len(averaged)),median,color='r')
    
    for i in Hist:
        plt.semilogy(np.arange(len(i)),i,alpha=0.1)
    plt.show()
        
    
    

def EndDist():
    #Number of nodes
    N = 10000
    
    #Proportion of zealots
    z = 0.6#0.3
    
    #Zealot strength
    F = 0.8#0.7
    
    #Time 
    T = 20
    
    Hist = []
    for Repeats in range(1):
        Hist.append(MainProcess(N, z, F,T,0.1)[2][:int(N*(1-z))])
        
        
        
    #Flatten
    all_values = np.concatenate(Hist)
        
    plt.figure()
    
    plt.hist(all_values,bins=100)
    plt.show()
    



def DistEvolution(FlexZealots = True):
    #Number of nodes
    N = 1000
    
    #Proportion of zealots
    z = 0.9#0.1
    
    #Zealot strength
    F = 0.2#0.99
    
    #Time 
    T = 1000000
    
    #Network
    Network = np.ones(N)
    
    """
    ###This starts the system off in the theoretical approach.
    p = F*z/(1-F+F*z)
    a = F*(1-p)/(1-F*p)
    
    Network[:int(N*p*(1-z))] = a
    
    
    #Set zealots
    Network[-int(N*z):] = a#F
    """

    Network[-int(N*z):] = F

    
    #Main Process
    meanlist = []
    maxlist = []
    minlist = []
    zealotlist = []
        
    checktimes = np.arange(0,T,T/1000)
    print(checktimes)
    
    # Define bin edges based on min/max opinions over all time steps
    num_bins = 1000
    opinion_min, opinion_max = 0, 1.1
    bin_edges = np.linspace(opinion_min, opinion_max, num_bins + 1)
    
    distribution_over_time = np.zeros((len(checktimes), num_bins))

    total_distribution_over_time = np.zeros(num_bins)
    total_distribution_over_time_nozealots = np.zeros(num_bins)
    
    i = 0
    for t in np.arange(T):
        if t in checktimes:
            counts, _ = np.histogram(Network, bins=bin_edges)  # Bin the opinions at time t
            counts_nozealots,_ = np.histogram(Network[:int(N*(1-z))],bins=bin_edges)
            distribution_over_time[i] = counts
            if t > T/2:
                total_distribution_over_time += counts
                total_distribution_over_time_nozealots += counts_nozealots
            i+=1
            print(t)
        
        #Choose index to update opinion
        updating_index = np.random.randint(0,N*(1-z)+1)
        
        # Choose an index with probability proportional to the values
        selected_index = np.random.choice(len(Network), p=Network / Network.sum())
    
        #Set the new value:
        Network[updating_index] = Network[selected_index]
        
        """
        if 1 not in Network:
            Network[0] = 1
        """
    
        #Calculate the new mean
        mean_nonzealots = np.mean(Network[:int(N*(1-z))])
        meanlist.append(mean_nonzealots)
        maxlist.append(max(Network))
        minlist.append(min(Network))
        zealotlist.append(Network[-1])
        
        #Update Zealots
        if FlexZealots:
            Network[-int(N*z):] = F* mean_nonzealots #*min(Network[:int(N*(1-z))])
        
        
    print(total_distribution_over_time)
    """
    distribution_over_time = np.transpose(distribution_over_time)
    
    #Entire histogram
    plt.figure()
    # Sum over time axis to get total frequency per bin
    total_counts = distribution_over_time[:, len(checktimes) // 2 :].sum(axis=1)  # Shape: (num_bins,)
    
    # Plot the histogram
    plt.figure(figsize=(10, 5))
    plt.bar(bin_edges[:-1], total_counts, width=np.diff(bin_edges), align='edge', color='royalblue', alpha=0.7)
    
    # Labels and title
    plt.xlabel("Opinion Value (Binned)")
    plt.ylabel("Total Frequency Over Time")
    plt.title("Collated Histogram of All Opinions Over Time")
    
    # Optional: Set log scale on Y-axis for better visibility
    plt.yscale("log")
    
    # Show plot
    plt.show()
    """
    
    print(bin_edges[-1])
    print("sum",sum(total_distribution_over_time_nozealots/(len(checktimes)/2)))
    plt.figure()
    #plt.bar(bin_edges[:-1],total_distribution_over_time,width=np.diff(bin_edges),align='edge', color='royalblue')
    plt.plot(bin_edges[:-1],total_distribution_over_time_nozealots/(N*len(checktimes)/2)/(1-z))
    plt.show()
        
                
    # Replace zero values with a small positive value (avoid log(0) issue)
    distribution_over_time[distribution_over_time == 0] = 1e-1   
    #popt = FitSlope(meanlist, T, SlopeCutoff)
    distribution_over_time = np.transpose(distribution_over_time)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(distribution_over_time, aspect='auto', cmap='viridis', origin='lower',
       extent=[0, T,bin_edges[0], bin_edges[-1]],
       norm=mcolors.LogNorm(vmin=np.min(distribution_over_time), vmax=np.max(distribution_over_time))) 
    
    # Add color bar
    cbar = fig.colorbar(im, ax=ax, label="Frequency (log scale)")
    cbar.ax.set_yscale('log')  # Ensure color bar also reflects log scale
    
    x = np.arange(len(meanlist))
    plt.plot(x,meanlist,label='mean')
    #plt.plot(x,meanfit(x, np.exp(popt[0]), popt[1]),linestyle='dashed',label='Mean fit')
    #plt.plot(x,maxlist,label='max')
    #plt.plot(x,zealotlist,label='zealots')
    plt.yscale("log")
    plt.ylabel("Opinion")
    plt.xlabel("Time")
    #plt.ylim(0.5,1.1)
    
    
    if FlexZealots:
        #Predicted average
        theory_mean = (1-F)/(1-F+z*F-z*F**2)
        def mean(a,p):
            return (a*p+1-p-z)/(1-z)
        
        
        p = F*z*(1-z)/(1-F+z*(2*F-1))
        
        a = F*(1-p-z)/(1-z-F*p)
        
        theory_mean = mean(a,p)
        print(mean(a, p))
            
        
        
        theory_mean = (1-z) / (1 - (1-F)*z)
        
    else:
        if z < (1-F):
            theory_mean = (z*F**2 + 1 - F - z)/((1-z)*(1-F))
        else:
            theory_mean = F
    
        print(theory_mean)
    
    plt.plot(x,theory_mean*np.ones(len(x)),linestyle='dashed')
    plt.title("F=%0.3f, z = %0.3f"%(F,z))
    
    
    
    
def CompareDists():
    #Number of nodes
    N = 10000
    
    #Proportion of zealots
    z = 0.1#0.1
    
    #Zealot strength
    F = 0.99#0.99
    
    #Time 
    T = 1000000
    
    #Network
    Network0 = np.ones(N)
    Network1 = np.ones(N)

    Network0[-int(N*z):] = F
    Network1[-int(N*z):] = F
    
    #Main Process
    meanlist0 = []
    meanlist1 = []
    maxlist = []
    minlist = []
    zealotlist = []
        
    checktimes = np.arange(0,T,T/1000)
    print(checktimes)
    mean_nonzealots0 = 1
    """
    # Define bin edges based on min/max opinions over all time steps
    num_bins = 1000
    opinion_min, opinion_max = 0, 1.1
    bin_edges = np.linspace(opinion_min, opinion_max, num_bins + 1)
    
    distribution_over_time = np.zeros((len(checktimes), num_bins))

    total_distribution_over_time = np.zeros(num_bins)
    total_distribution_over_time_nozealots = np.zeros(num_bins)
    """
    
    updatehist = np.zeros(len(Network0))
    
    i = 0
    for t in np.arange(T):
        if t in checktimes:
            """
            counts, _ = np.histogram(Network, bins=bin_edges)  # Bin the opinions at time t
            counts_nozealots,_ = np.histogram(Network[:int(N*(1-z))],bins=bin_edges)
            distribution_over_time[i] = counts
            if t > T/2:
                total_distribution_over_time += counts
                total_distribution_over_time_nozealots += counts_nozealots
            i+=1
            """
            print(t)
        
        #Choose index to update opinion
        updating_index = np.random.randint(0,N*(1-z)+1)
        updatehist[updating_index] +=1
        # Choose an index with probability proportional to the values
        selected_index0 = np.random.choice(len(Network0), p=Network0 / Network0.sum())
        selected_index1 = np.random.choice(len(Network1), p=Network1 / Network1.sum())
        #Set the new value:
        Network0[updating_index] = Network0[selected_index0]
        Network1[updating_index] = Network1[selected_index1]
        """
        if 1 not in Network:
            Network[0] = 1
        """
    
        #Calculate the new mean
        if mean_nonzealots0 == F:
            if np.mean(Network0[:int(N*(1-z))]) > 0:
                print(updating_index,selected_index0)
        
        mean_nonzealots0 = np.mean(Network0[:int(N*(1-z))])
        mean_nonzealots1 = np.mean(Network1[:int(N*(1-z))])
        meanlist0.append(mean_nonzealots0)
        meanlist1.append(mean_nonzealots1)
        
        #Update Zealots
        Network1[-int(N*z):] = F* mean_nonzealots1 #*min(Network[:int(N*(1-z))])
        
        
    """
    distribution_over_time = np.transpose(distribution_over_time)
    
    #Entire histogram
    plt.figure()
    # Sum over time axis to get total frequency per bin
    total_counts = distribution_over_time[:, len(checktimes) // 2 :].sum(axis=1)  # Shape: (num_bins,)
    
    # Plot the histogram
    plt.figure(figsize=(10, 5))
    plt.bar(bin_edges[:-1], total_counts, width=np.diff(bin_edges), align='edge', color='royalblue', alpha=0.7)
    
    # Labels and title
    plt.xlabel("Opinion Value (Binned)")
    plt.ylabel("Total Frequency Over Time")
    plt.title("Collated Histogram of All Opinions Over Time")
    
    # Optional: Set log scale on Y-axis for better visibility
    plt.yscale("log")
    
    # Show plot
    plt.show()
    """
    print(updatehist)
    #print(meanlist0)
    plt.figure()
    x = np.arange(len(meanlist0))
    plt.plot(x,meanlist0,label='mean0')
    plt.plot(x,meanlist1,label='mean1')
    #plt.plot(x,meanfit(x, np.exp(popt[0]), popt[1]),linestyle='dashed',label='Mean fit')
    #plt.plot(x,maxlist,label='max')
    #plt.plot(x,zealotlist,label='zealots')
    plt.yscale("log")
    plt.ylabel("Opinion")
    plt.xlabel("Time")
    #plt.ylim(0.5,1.1)
    
    
        
        
    theory_mean1 = (1-z) / (1 - (1-F)*z)
        
    if z < (1-F):
        theory_mean0 = (z*F**2 + 1 - F - z)/((1-z)*(1-F))
    else:
        theory_mean0 = F
    
    
    plt.plot(x,theory_mean0*np.ones(len(x)),linestyle='dashed',label='Theory 0')
    plt.plot(x,theory_mean1*np.ones(len(x)),linestyle='dashed',label='Theory 1')

    plt.title("F=%0.3f, z = %0.3f"%(F,z))
    
    plt.legend()
    """
    x = np.arange(len(meanlist))

    
    plt.plot(x,meanlist,label='mean')
    #plt.plot(x,meanfit(x, np.exp(popt[0]), popt[1]),linestyle='dashed',label='Mean fit')
    plt.plot(x,maxlist,label='max')
    plt.plot(x,minlist,label='min')
    plt.title("F=%0.3f, z = %0.3f"%(F,z))
    plt.legend()
    plt.yscale("log")
    plt.show()
    """


#EndDist()
Varyz()

#CompareDists()

endtime = time.time()

print("time",endtime-starttime)