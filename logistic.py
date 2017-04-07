import numpy as np
import pandas as pd
from sklearn import linear_model

# read in previously generated data
geno = pd.read_hdf('data/geno_small_corr.h5')
snps = pd.read_hdf('data/snps_small_corr.h5')

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
top_betas = abs(betas).sort_values()[-25:][::-1]
print(top_betas)