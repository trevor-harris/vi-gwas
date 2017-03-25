import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# parameters from the data creation
p = 100000
true_p = 5
n = 1500

# read in the ADVI samples from the Stan run
# advi_coef = pd.read_csv('/tmp/tmpnvp89ztb/output.csv'
# 	,skiprows = [0, 1, 2, 3, 5, 6])

# betas = advi_coef.filter(like = "beta")
# betas.to_hdf('/tmp/tmpnvp89ztb/output.h5', 'data', mode='w', format='fixed')

raw_betas = pd.read_hdf('/tmp/tmpnvp89ztb/output.h5')

# relabel to have the actual indicies of the betas after the split
raw_betas.columns = ["beta."+str(j) for j in range(0, p)]

print("MEDIANS")
betas = raw_betas.median(axis = 0)
top_betas = abs(betas).sort_values()[-25:][::-1]

print(np.arange(1, p, int(p/true_p)))
print(top_betas)

print("MEANS")
betas = raw_betas.mean(axis = 0)
top_betas = abs(betas).sort_values()[-25:][::-1]

print(np.arange(1, p, int(p/true_p)))
print(top_betas)