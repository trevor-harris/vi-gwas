import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats

raw_betas = pd.read_hdf('data/betas_small.h5')

n = raw_betas.shape[0]
p = raw_betas.shape[1]
true_p = 5

# print the actual values
actual = pd.read_hdf('data/actual_small_corr.h5')
print(actual)

# unadjusted
betas = raw_betas.median(axis = 0)
top_betas = abs(betas).sort_values()[-25:][::-1]

beta_index = list(top_betas.index)
top_betas = raw_betas.filter(items = beta_index)

sb.violinplot(data = top_betas)
sb.plt.xticks(rotation = 45)
sb.plt.show()

