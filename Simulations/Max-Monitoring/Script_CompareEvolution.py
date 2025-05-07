from Params import *

import sys
sys.path.insert(0,'../../CoreFunctions')

import Core


import matplotlib.pyplot as plt

import numpy as np

import copy

import time

starttime = time.time()




##########################
###Main Process###########
##########################

def P_Up(n,z,F):
    return (1-n-z)/(1-z) * F*(n+z) / (1-n-z+F*(n+z))


def P_Down(n,z,F):
    return n/(1-z) * (1-n-z) / (1-n-z+F*(n+z))



F = 0.93

T = 100000

nMatrix = []

for R in range(Repeats):
    print(R)

    nlist = []

    n = 0
    t = 0

    while True:

        prob = np.random.uniform(0,1)

        if prob < P_Up(n,z,F):
            n += 1/N

        elif prob < (P_Up(n,z,F) + P_Down(n,z,F)):
            n -= 1/N

        nlist.append(n)

        t += 1

        if abs(n - (1-z)) < 1/(2*N):
                break


    nMatrix.append(nlist)


# Find the maximum length
max_len = max(len(lst) for lst in nMatrix)

# Pad with 1s to make all lists the same length
padded = np.array([
    lst + [1-z] * (max_len - len(lst))
    for lst in nMatrix
])

# Compute mean along columns (axis=0)
means = padded.mean(axis=0)





#Analytical

nlist = np.linspace(0,1-z-1/(2*N)) 

a = 1-z
D = F*z/(1-F)
c = z - 1/(1-F)
#Exp = min(D,1-z)
#e = 1/N
k1= (F*np.log(1-z) - (1-z)*np.log(F*z/(1-F))) / (1-F-z)

t1 = a * (((c+D)*np.log(abs(nlist-D)) - (a+c) * np.log(abs(nlist-a)))/(a-D) - k1)







plt.figure()

for i in padded:
    plt.plot(np.arange(len(i)),i,alpha=0.7)

plt.plot(np.arange(len(means)),means)

plt.plot(t1*N,nlist,color='k',)
plt.savefig("Evolution_z_%0.3f_F_%0.3f_N_%d.png"%(z,F,N))
plt.close()
