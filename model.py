import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pystan import StanModel
from math import exp

# data shape
p = 100000
true_p = 5
n = 1500

# generate SNPs
maf = np.random.uniform(0.05, 0.5, p)
snps = np.empty(shape = (n, p))

for ind, f in np.ndenumerate(maf):
	snps[:, ind[0]] = np.random.binomial(2, f, n)

# normalize snps
snps = (snps - snps.mean(axis = 0)) / snps.std(axis = 0)

# generate genotypes
# select a subset to be in the true model
true_id = np.arange(1, p, int(p/true_p))
true_snps = snps[:, true_id]
true_beta = np.random.choice(np.array([-3, -2, -1, 1, 2, 3]), true_p)
true_alpha = -0.2

# create gene expression levels
def binomialize(x):
	p = 1 / (1 + exp(-x))
	return np.random.binomial(1, p, 1)
vbin = np.vectorize(binomialize)

geno = true_alpha + np.sum(true_snps * true_beta, axis = 1)
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
}

model {
	// construct horseshoe prior on the betas
	tau ~ cauchy(0, 0.0005);
	lambda ~ cauchy(0, 0.0005);
	for (p in 1:P)
	  beta[p] ~ normal(0, lambda[p] * tau);


	// instead use the laplace prior to regularize. Much better convergence.
	//beta ~ double_exponential(0, 0.5);

	// construct model 
	y ~ bernoulli_logit(alpha + x * beta);
}
'''
model = StanModel(model_code = gwas_code)

# ---------
# Mini-batch idea -- finds coeffs better (unless shrinkage is increased) but is much slower
# all_betas = pd.Series(0)
# splits = 100

# ts = time.time()
# for i in range(splits):

# 	sub_len = p // splits
# 	snps_i = snps[:, i * sub_len:(i + 1) * sub_len]

# 	gwas_data = {
# 		'N': n
# 		,'P': sub_len
# 		,'y': geno
# 		,'x': snps_i
# 	}

# 	# ADVI using stochastic(?) gradient descent
# 	fit_advi = model.vb(data = gwas_data
# 		,output_samples = 2000
# 		,iter = 10000
# 		,algorithm = 'meanfield')

# 	# read in the posterior draws for each parameter
# 	param_file = fit_advi['args']['sample_file'].decode("utf-8")
# 	advi_coef = pd.read_csv(param_file 
# 		,skiprows = [0, 1, 2, 3, 5, 6])

# 	betas = advi_coef.filter(like = "beta")

# 	# relabel to have the actual indicies of the betas after the split
# 	current_index = range(i * sub_len, (i + 1) * sub_len)
# 	betas.columns = ["beta."+str(j) for j in current_index]

# 	betas = betas.mean(axis = 0)
# 	top_betas = abs(betas).sort_values()[-1:]
# 	all_betas = all_betas.append(top_betas)

# print(np.arange(1, p, int(p/true_p)))
# print(all_betas.sort_values()[-25:][::-1])
# print(time.time() - ts_1)

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
advi_coef = pd.read_csv(param_file 
	,skiprows = [0, 1, 2, 3, 5, 6])

betas = advi_coef.filter(like = "beta")

# relabel to have the actual indicies of the betas after the split
betas.columns = ["beta."+str(j) for j in range(0, p)]

# save coefficients in hdf5 format
betas.to_hdf('/tmp/current_model/output.h5', 'data', mode='w', format='fixed')