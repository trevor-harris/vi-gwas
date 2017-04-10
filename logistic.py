import numpy as np
import pandas as pd
from sklearn import linear_model

# read in previously generated data
geno = pd.read_hdf('data/geno_small.h5')
snps = pd.read_hdf('data/snps_small.h5')

geno = geno.values.flatten()
snps = snps.values

n = snps.shape[0]
p = snps.shape[1]

regress = linear_model.LogisticRegression(warm_start = True)
betas = np.zeros(p)

# for i in range(p):
# 	fit = regress.fit(snps[:, i].reshape(-1, 1), geno)
# 	betas[i] = fit.coef_

# betas = pd.Series(betas)
# betas.index = ["beta." + str(j) for j in range(p)]

# top_betas = abs(betas).sort_values()[-25:][::-1]
# print(top_betas)


fit = regress.fit(snps, geno)

betas = fit.coef_
betas = pd.Series(betas.flatten())
betas.index = ["beta." + str(j) for j in range(p)]

top_betas = abs(betas).sort_values()[-15:][::-1]
top_index = list(top_betas.index)

print(betas.filter(items = top_index))

# print the actual values
actual = pd.read_hdf('data/actual_small_corr.h5')
print(actual)