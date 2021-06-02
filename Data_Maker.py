'''
Generates a data set for the model to learn from
'''

import random as random
from scipy.stats import gamma


def main():
    # Define parameters for the data creation
    n = 1000  # number of points to be created
    observed = 0.8  # percentage of points to be observations
    sources = 0.2  # percentage of points to be sources (with high values of s)
    prior = gamma(a=0.5, scale=0.2)
    def sources_dist(): return random.uniform(0.5, 1)  # function that generates values for sources

    assert (0 < observed < 1 and 0 <= sources < 1)
