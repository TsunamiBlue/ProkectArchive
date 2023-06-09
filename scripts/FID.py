import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

# calculate frechet inception distance
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


if __name__ == '__main__':
    # define two collections of activations, testing
    act1 = random(10*2048)
    act1 = act1.reshape((10,2048))

    act2 = random(10*2048)
    act2 = act2.reshape((10,2048))

    # normal1 = 1/2 * np.normal
    print(act1)

    # fid between two
    fid = calculate_fid(act1, act2)
    print('FID: %.3f' % fid)
