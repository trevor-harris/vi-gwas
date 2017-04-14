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

	// horseshoe prior parameters
	vector<lower = 0>[P] lambda;
	vector<lower = 0>[P] eta;
	real<lower = 0> tau;

	// laplace parameters
	//real<lower = 0> scale;
}

model {
	tau ~ cauchy(0, 1);
	eta ~ cauchy(0, 1);
	lambda ~ cauchy(0, tau * eta);
	beta ~ normal(0, lambda);

	// instead use the laplace prior to regularize. Much easier convergence.
	//scale ~ gamma(2, 0.1);
	//scale ~ cauchy(0, 1);
	//scale ~ student_t(4, 0, 1);

	//scale ~ student_t(1, 0, 4);
	//beta ~ double_exponential(0, scale);

	// construct model 
	y ~ bernoulli_logit(x * beta);
}
'''
model = StanModel(model_code = gwas_code)

# read in previously generated data
geno = pd.read_hdf('data/geno.h5')
snps = pd.read_hdf('data/snps.h5')

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

# # HMC 
# fit_hmc = model.sampling(data = gwas_data, n_jobs = -1)
# print(fit_hmc)

# ADVI using stochastic gradient descent
fit_advi = model.vb(data = gwas_data
	,output_samples = 1000
	,iter = 40000
	,eval_elbo = 250
	,tol_rel_obj = 0.01
	,algorithm = 'meanfield')

# read in the posterior draws for each parameter
param_file = fit_advi['args']['sample_file'].decode("utf-8")
advi_coef = pd.read_csv(param_file, skiprows = [0, 1, 2, 3, 5, 6])

# relabel to have the actual indicies of the betas after the split
betas = advi_coef.filter(like = "beta")
betas.columns = ["beta." + str(j) for j in range(0, p)]

# save coefficients in hdf5 format
betas.to_hdf('data/betas.h5', 'data', mode='w', format='fixed')

# save other estimates in hdf5
param = advi_coef.filter(regex = "^(?!beta).*$")
param.to_hdf('data/param.h5', 'data', mode='w', format='fixed')
