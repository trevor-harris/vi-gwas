import numpy as np
import pandas as pd
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
	vector[P] beta;
	real<lower = 0> scale;

	// horseshoe prior parameters
	//vector<lower = 0>[P] lambda;
	//vector<lower = 0>[P] eta;
	//real<lower = 0> tau;
}

model {
	//tau ~ cauchy(0, 0.1);
	//lambda ~ cauchy(0, tau);
	//beta ~ normal(0, lambda);

	// instead use the laplace prior to regularize. Much easier convergence.
	//scale ~ gamma(2, 0.1);
	beta ~ double_exponential(0, scale);

	// construct model 
	y ~ bernoulli_logit(x * beta);
}
'''
model = StanModel(model_code = gwas_code)

# read in previously generated data
geno = pd.read_hdf('data/geno_small_corr.h5')
snps = pd.read_hdf('data/snps_small_corr.h5')

geno = geno.values.flatten()
snps = snps.values

n = snps.shape[0]
p = snps.shape[1]

gwas_data = {
		'N': n
		,'P': p
		,'y': geno
		,'x': snps
		,'shrink': 0.001
	}

# # HMC 
# fit_hmc = model.sampling(data = gwas_data, n_jobs = -1)
# print(fit_hmc)

# ADVI using stochastic(?) gradient descent
fit_advi = model.vb(data = gwas_data
	,output_samples = 1000
	,iter = 10000
	,eval_elbo = 50
	,tol_rel_obj = 0.01
	,algorithm = 'meanfield')

# read in the posterior draws for each parameter
param_file = fit_advi['args']['sample_file'].decode("utf-8")
advi_coef = pd.read_csv(param_file, skiprows = [0, 1, 2, 3, 5, 6])

# relabel to have the actual indicies of the betas after the split
betas = advi_coef.filter(like = "beta")
betas.columns = ["beta." + str(j) for j in range(0, p)]

# save coefficients in hdf5 format
betas.to_hdf('data/betas_small.h5', 'data', mode='w', format='fixed')