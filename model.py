import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from pystan import StanModel
from math import exp

# data shape
p = 3000
true_p = 10
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

	// horseshoe+ prior parameters
	//vector<lower = 0>[P] lambda;
	//vector<lower = 0>[P] eta;
	//real<lower = 0> tau;
}

model {
	// construct horseshoe+ prior on the betas...this has to be wrong..
	//tau ~ cauchy(0, 1);
	//eta ~ cauchy(0, 1);
	//lambda ~ cauchy(0, 1);
	//beta ~ normal(0, lambda .* eta * tau);

	// instead use the laplace prior to regularize. Much better convergence.
	beta ~ double_exponential(0, 1);

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

# ADVI using stochastic(?) gradient descent
fit_advi = model.vb(data = gwas_data
	,output_samples = 2000
	,iter = 10000
	,algorithm = 'meanfield')

# read in the posterior draws for each parameter
param_file = fit_advi['args']['sample_file'].decode("utf-8")
advi_coef = pd.read_csv(param_file 
	,skiprows = [0, 1, 2, 3, 5, 6]
	)

print(param_file)

# betas = advi_coef.filter(like = "beta")
# betas = betas[betas.apply(lambda x: )]

# # note that the beta number is snps number + 1
print(true_beta)
# sb.boxplot(data = advi_coef.filter(like = "beta"), showfliers=False)
# sb.plt.show()

betas = advi_coef.filter(like = "beta")
betas = betas.mean(axis = 0)
print(np.arange(1, p, int(p/true_p)) + 1)
print(abs(betas).sort_values()[-20:])

