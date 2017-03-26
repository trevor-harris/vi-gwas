import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sb

raw_betas = pd.read_hdf(os.path.expanduser('~/research/vi-gwas/data/betas.h5'))
n = raw_betas.shape[0]
p = raw_betas.shape[1]
true_p = 5

print("MEDIANS")
betas = raw_betas.median(axis = 0)
top_betas = abs(betas).sort_values()[-25:][::-1]

print(np.arange(1, p, int(p/true_p)))
print(top_betas)
