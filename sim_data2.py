# generate the GWAS data for testing the variationl bayes methods
import numpy as np
import pandas as pd
from math import exp


# data shape
p = 5000
true_p = 10
n = 1000

# make matrix a tridiagonal matrix
def tridiag(mat, k1=-1, k2=0, k3=1):
	a = np.diag(mat, k1)
	b = np.diag(mat, k2)
	c = np.diag(mat, k3)
	return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

# generate mean
mu = np.ones(p)

# generate covariance
sig = (np.random.rand(p, p) - np.random.rand(p, p))**2
sig = np.dot(sig, np.transpose(sig))
sig = tridiag(sig, -2, -1, 0) + tridiag(sig, 0, 1, 2)

# generate snps with the above mean and corr structure
snps = np.random.multivariate_normal(mu, sig, size=n)
# snps = (snps - snps.mean(axis = 0)) / snps.std(axis = 0)

# generate genotypes
# select a subset to be in the true model
# true_id = np.arange(1, p, int(p/true_p))
true_id = np.arange(1, true_p + 1)
true_snps = snps[:, true_id]
true_beta = np.random.choice(np.array([-0.4, -0.3, -0.2, 0.2, 0.3, 0.4]), true_p)

# create gene expression levels
def binomialize(x):
	p = 1 / (1 + exp(-x))
	return np.random.binomial(1, p, 1)
vbin = np.vectorize(binomialize)

true_snps = np.array(true_snps)
true_beta = np.array(true_beta)

# geno = true_alpha + np.sum(np.dot(true_snps, true_beta), axis = 1)
geno = np.sum(true_snps * true_beta, axis = 1)
geno = vbin(geno)

geno = pd.DataFrame(geno)
snps = pd.DataFrame(snps)

# write the data in hdf5 format for faster reading
snps.to_hdf('data/snps_small_corr.h5', 'data', mode='w', format='fixed')
geno.to_hdf('data/geno_small_corr.h5', 'data', mode='w', format='fixed')

# save the true parameters
actual = pd.DataFrame({
	'snp_id': true_id
	,'beta': true_beta 
	})

actual.to_hdf('data/actual_small_corr.h5', 'data', mode='w', format='fixed')
