# generate the GWAS data for testing the variationl bayes methods
import numpy as np
import pandas as pd
import os
from math import exp

# data shape
p = 500000
true_p = 5
n = 2000

# generate SNPs
maf = np.random.uniform(0.05, 0.5, p)
snps = np.empty(shape = (n, p))

for ind, f in np.ndenumerate(maf):
	snps[:, ind[0]] = np.random.binomial(2, f, n)

# normalize snps
snps = (snps - snps.mean(axis = 0)) / snps.std(axis = 0)

# generate genotypes
# select a subset to be in the true model
true_id = np.arange(1, p, int(p/true_p))
true_snps = snps[:, true_id]
true_beta = np.random.choice(np.array([-3, -2, -1, 1, 2, 3]), true_p)
true_alpha = -0.2

# create gene expression levels
def binomialize(x):
	p = 1 / (1 + exp(-x))
	return np.random.binomial(1, p, 1)
vbin = np.vectorize(binomialize)

geno = true_alpha + np.sum(true_snps * true_beta, axis = 1)
geno = vbin(geno)

geno = pd.DataFrame(geno)
snps = pd.DataFrame(snps)

# write the data in hdf5 format for faster reading
home = os.path.expanduser('~')
snps.to_hdf(home + '/research/vi-gwas/data/snps.h5', 'data', mode='w', format='fixed')
geno.to_hdf(home + '/research/vi-gwas/data/geno.h5', 'data', mode='w', format='fixed')