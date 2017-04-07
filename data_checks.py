import numpy as np
import pandas as pd


# read in previously generated data
snps = pd.read_hdf('data/snps_small_corr.h5')

# print(snps.head())

print(snps.iloc[:, 1:10].corr())

