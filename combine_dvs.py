import sys, platform, os
#sys.path.insert(0, os.environ['ROOTDIR'] + 
#                   '/external_modules/code/CAMB/build/lib.linux-x86_64-'
#                   +os.environ['PYTHON_VERSION'])

from os.path import join as pjoin
import numpy as np


training_sample_dir = '/groups/behroozi/hbowden/cocoa_gpu/Cocoa/projects/roman/emulator_output/training_sample/'

lhs0 = np.loadtxt(training_sample_dir+'dump/lhs_1000_0-4_cs.txt',delimiter=' ')
lhs1 = np.loadtxt(training_sample_dir+'dump/lhs_1000_1-4_cs.txt',delimiter=' ')
lhs2 = np.loadtxt(training_sample_dir+'dump/lhs_1000_2-4_cs.txt',delimiter=' ')
lhs3 = np.loadtxt(training_sample_dir+'dump/lhs_1000_3-4_cs.txt',delimiter=' ')

dvs = np.concatenate([lhs0,lhs1,lhs2,lhs3])

params = dvs[:,0:8]
data_vectors = dvs[:,8:]

print(np.shape(params))
print(np.shape(data_vectors))

np.save(training_sample_dir+'data_vectors_test_1000.npy',data_vectors)
np.save(training_sample_dir+'params_test_1000.npy',params)