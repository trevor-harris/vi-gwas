import numpy as np
import pandas as pd
from math import exp


# read in previously generated data
geno = pd.read_hdf('data/geno_small_corr.h5')
snps = pd.read_hdf('data/snps_small_corr.h5')

geno = geno.values.flatten()
snps = snps.values

n = snps.shape[0]
p = snps.shape[1]

print(snps.shape)
print(geno.shape) # should be (1000, )
