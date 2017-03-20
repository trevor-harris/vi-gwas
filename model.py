import numpy as np
from pystan import StanModel
from math import exp

# data shape
p = 1000
true_p = 10
n = 500

# generate SNPs
maf = np.random.uniform(0.05, 0.5, p)
snps = np.empty(shape = (n, p))

for ind, f in np.ndenumerate(maf):
	snps[:, ind[0]] = np.random.binomial(2, f, n)

# generate genotypes
# select a subset to be in the true model
true_snps_ind = np.random.choice(np.arange(1, p), true_p, replace = False)
print(true_snps_ind)

true_snps = snps[:, true_snps_ind]
true_beta = np.random.normal(0, 1, true_p)

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
	//vector<lower = 0, upper = 1>[N] y;
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
fit_advi = model.vb(data = gwas_data)
print(fit_advi['args']['sample_file'])
# print(fit_advi)

# extract the model coefficients
# beta = np.mean(fit_advi.extract()['beta'], axis=0)
# print(beta)


# HMC using NUTS
# fit_hmc = model.sampling(data = gwas_data, iter = 1000, chains = 4)

# extract the model coefficients
# beta = np.mean(fit_hmc.extract()['beta'], axis=0)
# print(beta)







