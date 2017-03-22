import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pystan import StanModel
from math import exp

# data shape
p = 300
true_p = 5
n = 1000


# read in the posterior draws for each parameter
param_file = "/tmp/tmpo3u2spqi/output.csv"
advi_coef = pd.read_csv(param_file 
	,skiprows = [0, 1, 2, 3, 5, 6]
	)

print(np.arange(1, p, int(p/true_p)) + 1)

betas = advi_coef.filter(like = "beta")
betas = betas.mean(axis = 0)
print(abs(betas).sort_values()[-10:])

# print(sorted(betas.argsort()[-10:][::-1]))

# sb.tsplot(data = betas, interpolate = False)
# # sb.distplot(betas['beta.51'])
# sb.plt.show()
# print(betas.mean(axis = 0))

# top_betas = np.where(abs(betas) > 0.9)
# print(top_betas)