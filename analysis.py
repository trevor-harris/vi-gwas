import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy import stats

raw_betas = pd.read_hdf('data/betas_small.h5')

n = raw_betas.shape[0]
p = raw_betas.shape[1]
true_p = 5

print("MEDIANS")
betas = raw_betas.median(axis = 0)
top_betas = abs(betas).sort_values()[-25:][::-1]

# print(np.arange(1, p, int(p/true_p)))
# print(top_betas)

beta_index = list(top_betas.index)
top_betas = raw_betas.filter(items = beta_index)

sb.violinplot(data = abs(top_betas))
sb.plt.xticks(rotation = 45)
sb.plt.show()