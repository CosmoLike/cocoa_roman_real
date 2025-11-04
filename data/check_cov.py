#Make sure covariance matrix with given mask is positive definite

import numpy as np
import sys
import os

covfile = sys.argv[1]
maskfile = sys.argv[2]

def get_mask(maskfile):
    mask = np.genfromtxt(maskfile, delimiter=' ')
    #print("Mask shape:", mask.shape)
    mask = mask[:, 1]  # Extract the second column which contains the mask values
    return mask


def check_covariance(covfile, maskfile):
    data = np.genfromtxt(covfile)
    ndata = int(np.max(data[:,0]))+1
    cov_g = np.zeros((ndata,ndata))
    cov_ng = np.zeros((ndata,ndata))
    for i in range(0,data.shape[0]):
        cov_g[int(data[i,0]),int(data[i,1])] =data[i,8]
        cov_g[int(data[i,1]),int(data[i,0])] =data[i,8]
        cov_ng[int(data[i,0]),int(data[i,1])] =data[i,9]
        cov_ng[int(data[i,1]),int(data[i,0])] =data[i,9]
    cov = cov_g + cov_ng
    
    mask = get_mask(maskfile)

    # Apply mask to covariance matrix
    #masked_cov = cov[mask][:, mask]
    locs = np.where(mask==0)
    unneeded = np.r_[locs[0]]
    masked_cov = np.delete(cov,unneeded,0)
    masked_cov = np.delete(masked_cov,unneeded,1)

    # Check if the masked covariance matrix is positive definite
    try:
        np.linalg.cholesky(masked_cov)
        print("Covariance matrix is positive definite.")
    except np.linalg.LinAlgError:
        print("Covariance matrix is not positive definite.")

    print("Covariance matrix shape:", cov.shape)
    print("Masked covariance matrix shape:", masked_cov.shape)
    return len(cov)

cov_size = check_covariance(covfile, maskfile)

filename = covfile.removesuffix('_cov')+'.modelvector'
if os.path.exists(filename):
    # If data vector exists, skip
    print(f"Data vector {filename} already exists. Skipping creation.")
else:
    # If data vector does not exist, create it
    print(f"Data vector {filename} does not exist. Creating it now.")
    data_vector = np.zeros((cov_size,2))
    data_vector[:, 0] = np.arange(cov_size)  # First column: indices
    data_vector[:, 1] = 0.0  # Second column: zeros
    np.savetxt(filename, data_vector, fmt='%.6f')
