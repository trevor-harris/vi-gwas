import numpy as np
import pandas as pd
from sklearn import linear_model

# read in previously generated data
geno = pd.read_hdf('data/geno.h5')
snps = pd.read_hdf('data/snps.h5')

geno = geno.values.flatten()
snps = snps.values

n = snps.shape[0]
p = snps.shape[1]

regress = linear_model.LogisticRegression(warm_start = True, penalty = 'l1')
fit = regress.fit(snps, geno)

betas = np.zeros(p)
betas = fit.coef_
betas = pd.Series(betas.flatten())
betas.index = ["beta." + str(j) for j in range(p)]

top_betas = abs(betas).sort_values()[-100:][::-1]
top_index = list(top_betas.index)

print(betas.filter(items = top_index))

# print the actual values
actual = pd.read_hdf('data/actual.h5')
print(actual)