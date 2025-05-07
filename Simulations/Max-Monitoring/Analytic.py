import sys
sys.path.insert(0,'../../CoreFunctions')

import Core

import matplotlib as mpl
mpl.use("Agg")


import matplotlib.pyplot as plt

import numpy as np

import copy

import time

starttime = time.time()


N = 1000


#Functions
def P_Up(n,z,F):
    return (1-n-z)/(1-z) * F*(n+z) / (1-n-z+F*(n+z))


def P_Down(n,z,F):
    return n/(1-z) * (1-n-z) / (1-n-z+F*(n+z))




def solve_tridiagonal(a, b, c, d):
    """
    Solve a tridiagonal system Ax = d where:
    - a: sub-diagonal (length n-1)
    - b: main diagonal (length n)
    - c: super-diagonal (length n-1)
    - d: right-hand side (length n)
    Returns x (solution to Ax = d)
    """
    n = len(b)
    cp = np.zeros(n-1)
    dp = np.zeros(n)

    # Forward sweep
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n-1):
        denom = b[i] - a[i-1] * cp[i-1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i-1] * dp[i-1]) / denom
    dp[n-1] = (d[n-1] - a[n-2] * dp[n-2]) / (b[n-1] - a[n-2] * cp[n-2])

    # Back substitution
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n-2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i+1]

    return x



def compute_mfpt_fast(p_up_func, p_down_func, N, z,F):
    M = int(N * (1 - z))  # number of states before absorbing boundary

    a = np.zeros(M - 1)  # sub-diagonal (p_down)
    b = np.zeros(M)      # main diagonal
    c = np.zeros(M - 1)  # super-diagonal (p_up)
    d = np.ones(M)       # right-hand side (all ones)

    for i in range(M):
        ni = i / N
        pup = p_up_func(ni,z,F)
        pdown = p_down_func(ni,z,F)

        b[i] = pup + pdown
        if i > 0:
            a[i - 1] = -pdown  # sub-diagonal: connects T[i] to T[i-1]
        if i < M - 1:
            c[i] = -pup        # super-diagonal: connects T[i] to T[i+1]

    T = solve_tridiagonal(a, b, c, d)
    return T[0]  # MFPT starting from state n = 0



#Params are N, z
ParamsList = [
        (1000,0.1),
        (10000,0.1),
        (1000,0.05)]

zList = [0.01,0.05,0.1]

FMatrix = []
FirstPassageTimeMatrix = []

for i in ParamsList:
    N,z = i

    FirstPassageTimeList = []

    FList = np.linspace(1-1.3*z,1-0.0001,100)  #Broaden lower range to highlight

    if z == 0.05:
        FList = np.linspace(0.9,1-0.001,100)

    FMatrix.append(FList)

    for F in FList:

        FirstPassageTimeList.append(compute_mfpt_fast(P_Up, P_Down, N, z,F))


    FirstPassageTimeMatrix.append(np.asarray(FirstPassageTimeList))



#Plottingi
width = 40 / 25.4
fig, ax = plt.subplots(figsize=(2*width, 30 / 25.4))

for i in range(len(FirstPassageTimeMatrix)):
    if i == 0:
        linestyle = 'solid'
    elif i == 1:
        linestyle = 'dashed'
    elif i == 2:
        linestyle = 'dotted'


    plt.semilogy(FMatrix[i],-FirstPassageTimeMatrix[i]/np.log(FMatrix[i]),linestyle=linestyle,color='k',linewidth=2)


ax.set_xticks([0.9,0.95,1])
ax.set_xticklabels([r'$0.90$',r'$0.95$',r'$1.00$'],fontsize=7)

ax.set_yticks([10**6,10**7,10**8,10**9])
ax.set_yticklabels([r'$10^6$',r'$10^7$',r'$10^8$',r'$10^9$'],fontsize=7)

plt.xticks(fontname='Arial')

plt.ylim(1e6,10**(9.5))
plt.xlim(0.9,1)
plt.savefig("Compares.png",bbox_inches='tight', dpi=300)
plt.close()









N = 1000

zList = np.linspace(0.01,0.99,100)

Min_FList = []
Min_TimeList = []

for z in zList:

    print(z)
    FList = np.linspace(1-z,1-0.001,100)

    
    FirstPassageTimeList = []

    for F in FList:
        FirstPassageTimeList.append(compute_mfpt_fast(P_Up, P_Down, N, z,F))

    TotalTime = -np.asarray(FirstPassageTimeList)/np.log(FList)

    mindex = np.argmin(TotalTime)

    Min_FList.append(FList[mindex])
    Min_TimeList.append(TotalTime[mindex])



fig, ax = plt.subplots(figsize=(width, 30 / 25.4))
plt.plot(zList,Min_FList,linewidth=2,color='k')


ax.set_xticks([0,0.5,1])
ax.set_xticklabels([r'$0.0$',r'$0.5$',r'$1.0$'],fontsize=7)

ax.set_yticks([0,0.5,1])
ax.set_yticklabels([r'$0.0$',r'$0.5$',r'$1.0$'],fontsize=7)
plt.xticks(fontname='Arial')

plt.ylim(0,1)
plt.xlim(0,1)

plt.savefig("MinFg.png",bbox_inches='tight', dpi=300)
plt.close()

################################################
fig, ax = plt.subplots(figsize=(width, 30 / 25.4))
plt.semilogy(zList,Min_TimeList,linewidth=2,color='k')

ax.set_xticks([0,0.2,0.4,0.6,0.8,1])
ax.set_xticklabels([r'$0.0$',r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'],fontsize=7)

ax.set_yticks([10**0,10**2,10**4,10**6,10**8])
ax.set_yticklabels([r'$10^0$',r'$10^2$',r'$10^4$',r'$10^6$',r'$10^8$'],fontsize=7)
plt.xticks(fontname='Arial')

plt.ylim(1,1e8)
plt.xlim(0,1)

plt.savefig("MinTime.png",bbox_inches='tight', dpi=300)


