# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sb

# read in the posterior draws for each parameter
# advi_coef = pd.read_csv('~/research/vi-gwas/parameters.csv'
# 	,skiprows = [0, 1, 2, 3, 5, 6]
# 	)

# sb.violinplot(data = advi_coef.filter(like = "beta"))
# sb.plt.show()
# betas = advi_coef.stack()
# print(betas)

# beta_hat = advi_coef.mean(axis = 0)[:102]
# print(beta_hat)

# sb.tsplot(beta_hat)
# sb.plt.show()

# sb.distplot(advi_coef['beta.8'])
# sb.plt.show()
