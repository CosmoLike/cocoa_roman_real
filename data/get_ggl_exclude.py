# Read order file to find which ggl bin combos are excluded in covariance matrix

import os
import sys
import numpy as np
import pandas as pd

filename = sys.argv[1]

N_LENS = 8  # Number of lens tomographic bins
N_SRC = 8   # Number of source tomographic bins

def get_Nggl(filename):
    order = np.genfromtxt(filename,dtype=['i8','f8','<U5','<U5','<U5'],
                     delimiter=' ',comments='#',names=['idx','val','type','bin1','bin2'])
    order = pd.DataFrame(order)
    gamma = order[order['type']=='gamma']
    gamma_combos = [(int(gamma['bin1'].iloc[i][1:]),int(gamma['bin2'].iloc[i][1:])) for i in range(len(gamma))]
    N = np.zeros((N_LENS,N_SRC))
    for i in np.arange(0,N_LENS):
        for j in np.arange(0,N_SRC):
            if (i,j) in gamma_combos:
                N[i][j] = 1
            else:
                N[i][j] = 0
    return N

ggl_exclude = get_Nggl(filename)

print("GGL Exclude:")
print(ggl_exclude)

