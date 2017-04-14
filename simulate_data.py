# generate the GWAS data for testing the variationl bayes methods
import numpy as np
import pandas as pd
from math import exp


# data shape
p = 100
true_p = 5
n = 5000

maf = np.random.uniform(0.05, 0.5, p)
snps = np.empty(shape = (n, p))

for ind, f in np.ndenumerate(maf):
	snps[:, ind[0]] = np.random.binomial(2, f, n)

# normalize snps
snps = (snps - snps.mean(axis = 0)) / snps.std(axis = 0)

# generate genotypes
# select a subset to be in the true model
true_id = np.arange(1, true_p + 1)
true_snps = snps[:, true_id]
true_beta = np.random.normal(20, 1, len(true_id))
print(true_beta)

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
snps.to_hdf('data/snps.h5', 'data', mode='w', format='fixed')
geno.to_hdf('data/geno.h5', 'data', mode='w', format='fixed')

# save the true parameters
actual = pd.DataFrame({
	'snp_id': true_id
	,'beta': true_beta 
	})

actual.to_hdf('data/actual.h5', 'data', mode='w', format='fixed')
