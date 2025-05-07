from Params import *

import sys
sys.path.insert(0,'../../CoreFunctions')

import Core


import matplotlib.pyplot as plt

import numpy as np

import copy

import time

starttime = time.time()


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



def compute_mfpt_fast(p_up_func, p_down_func, N, z):
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


"""
def ODE(t,n):
    a = 1-z
    D = F*z/(1-F)
    c = z - 1/(1-F)

    dndt = (a-n) * (n-D)/(a * (c+n))
            
    return dndt
"""
FList = np.asarray([0.93,0.94,0.95,0.96,0.97,0.98,0.99])

t1_AnalyticList = []
t1_NumericList = []

for F in FList:
    print("F:",F)
    #Analytic

    """
    k1= (F*np.log(1-z) - (1-z)*np.log(F*z/(1-F))) / (1-F-z)

    a = 1-z
    D = F*z/(1-F)
    c = z - 1/(1-F)
    Exp = min(D,1-z)
    e = 1/N 
    t1 = a * (((c+D)*np.log(abs(Exp-e-D)) - (a+c) * np.log(abs(Exp-e-a)))/(a-D) - k1)


    t1_AnalyticList.append(t1*N)

    #t1 = (1-z)/ (1-z-F) * (F * np.log(1/(N*(1-z))) - (1-z) * np.log(abs((1-F)/(F*z) * (1-z-1/N)-1)))
    """

    t1 = compute_mfpt_fast(P_Up, P_Down, N, z)
    t1_AnalyticList.append(t1)
    print("expected:",t1)

    tlist = []

    for r in range(Repeats):
        print("Repeat",r)
        n = 0
        t = 0

        #Main process
        while True:
            prob = np.random.uniform(0,1)

            if prob < P_Up(n,z,F):
                n += 1/N

            elif prob < (P_Up(n,z,F) + P_Down(n,z,F)):
                n -= 1/N


            t += 1
            #Stop when maximum is reached
            if abs(n - (1-z)) < 1/(2*N):
                break


        tlist.append(t)
        print(t)

    print("Mean:",np.mean(tlist))
    print("Median:",np.median(tlist))

    t1_NumericList.append(np.mean(tlist))

    endtime = time.time()

    print("Time taken:",endtime-starttime)


plt.figure()
plt.plot(FList,np.asarray(t1_AnalyticList))#/np.log(FList))
plt.plot(FList,np.asarray(t1_NumericList))#/np.log(FList))
plt.savefig("test_MeanFirstPassageTime.png")
