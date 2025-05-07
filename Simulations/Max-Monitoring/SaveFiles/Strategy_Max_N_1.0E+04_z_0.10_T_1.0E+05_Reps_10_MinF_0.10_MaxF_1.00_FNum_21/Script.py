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
    k1= (F*np.log(1-z) - (1-z)*np.log(F*z/(1-F))) / (1-F-z)

    a = 1-z
    D = F*z/(1-F)
    c = z - 1/(1-F)
    Exp = min(D,1-z)
    e = 1/N 
    t1 = a * (((c+D)*np.log(abs(Exp-e-D)) - (a+c) * np.log(abs(Exp-e-a)))/(a-D) - k1)


    t1_AnalyticList.append(t1*N)

    #t1 = (1-z)/ (1-z-F) * (F * np.log(1/(N*(1-z))) - (1-z) * np.log(abs((1-F)/(F*z) * (1-z-1/N)-1)))

    print("expected:",t1*N)

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

    t1_NumericList.append(np.median(tlist))

    endtime = time.time()

    print("Time taken:",endtime-starttime)


plt.figure()
plt.plot(FList,np.asarray(t1_AnalyticList))#/np.log(FList))
plt.plot(FList,np.asarray(t1_NumericList))#/np.log(FList))
plt.savefig("test_e_%d.png"%(e*N))
