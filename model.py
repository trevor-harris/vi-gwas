import numpy as np
import pandas as pd
import os
from pystan import StanModel

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
}

model {
	// construct horseshoe prior on the betas
	tau ~ cauchy(0, 0.0001);
	lambda ~ cauchy(0, 0.0001);
	beta ~ normal(0, lambda * tau);
	//for (p in 1:P)
	//  beta[p] ~ normal(0, lambda[p] * tau);


	// instead use the laplace prior to regularize. Much easier convergence.
	//beta ~ double_exponential(0, 0.5);

	// construct model 
	y ~ bernoulli_logit(alpha + x * beta);
}
'''
model = StanModel(model_code = gwas_code)

# read in previously generated data
home = os.path.expanduser('~')
geno = pd.read_hdf(home + '/research/vi-gwas/data/geno_small.h5')
snps = pd.read_hdf(home + '/research/vi-gwas/data/snps_small.h5')

geno = geno.values.flatten()
snps = snps.values

n = snps.shape[0]
p = snps.shape[1]

gwas_data = {
		'N': n
		,'P': p
		,'y': geno
		,'x': snps
	}

# ADVI using stochastic(?) gradient descent
fit_advi = model.vb(data = gwas_data
	,output_samples = 1000
	,iter = 10000
	,algorithm = 'meanfield')

# read in the posterior draws for each parameter
param_file = fit_advi['args']['sample_file'].decode("utf-8")
advi_coef = pd.read_csv(param_file, skiprows = [0, 1, 2, 3, 5, 6])

# relabel to have the actual indicies of the betas after the split
betas = advi_coef.filter(like = "beta")
betas.columns = ["beta." + str(j) for j in range(0, p)]

# save coefficients in hdf5 format
betas.to_hdf(home + '/research/vi-gwas/data/betas_small.h5', 'data', mode='w'
	,format='fixed')