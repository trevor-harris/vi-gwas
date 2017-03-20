import numpy as np
from pystan import StanModel
from math import exp

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# data shape
p = 100
true_p = 10
n = 100

# generate SNPs
maf = np.random.uniform(0.05, 0.5, p)
snps = np.empty(shape = (n, p))

for ind, f in np.ndenumerate(maf):
	snps[:, ind[0]] = np.random.binomial(2, f, n)

# normalize snps
snps = (snps - snps.mean(axis = 0)) / snps.std(axis = 0)

# generate genotypes
# select a subset to be in the true model
true_snps_ind = np.random.choice(np.arange(1, p), true_p, replace = False)
true_snps = snps[:, true_snps_ind]
true_beta = np.random.normal(0, 1, true_p)

# TODO: make this into an actual df and save it
print(true_snps_ind)
print(true_beta)

# create gene expression levels
def binomialize(x):
	p = 1 / (1 + exp(-x))
	return np.random.binomial(1, p, 1)
vbin = np.vectorize(binomialize)

geno = true_snps * true_beta
geno = np.sum(geno, axis = 1)
geno = vbin(geno)

# Begin Stan section
gwas_code = '''
data {
	// dimensions of the data
	int<lower = 0> N;
	int<lower = 0> P;

	// predictors (x) and target (y)
	matrix[N, P] x;
	int<lower = 0, upper = 1> y[N];
}

parameters {
	// regression coefficients
	real alpha;
	vector[P] beta;

	// horseshoe prior parameters
	vector<lower = 0>[P] lambda;
	real<lower = 0> tau;

	// epsilon parameters
	real<lower = 0> sigma; //std
}

model {
	// construct horseshoe prior on the betas
	lambda ~ cauchy(0, 1);
	tau ~ cauchy(0, 1);
	beta ~ normal(0, lambda * tau);

	//for(p in 1:P)
	//  beta[p] ~ normal(0, lambda[p] * tau);

	// construct model 
	y ~ bernoulli_logit(alpha + x * beta);
}
'''

gwas_data = {
	'N': n
	,'P': p
	,'y': geno
	,'x': snps
}

model = StanModel(model_code = gwas_code)

# ADVI using Stochastic gradient descent
fit_advi = model.vb(data = gwas_data
	,output_samples = 2000
	,iter = 20000
	,algorithm = 'meanfield')


# read in the posterior draws for each parameter
param_file = fit_advi['args']['sample_file'].decode("utf-8")
advi_coef = pd.read_csv(param_file 
	,skiprows = [0, 1, 2, 3, 5, 6]
	)

# note that the beta number is snps number + 1
# TODO: definitely need a more efficient plot for this many parameters...
sb.violinplot(data = advi_coef.filter(like = "beta"))
sb.plt.show()
