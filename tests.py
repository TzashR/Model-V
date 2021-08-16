import random
from scipy.stats import gamma
import matplotlib.pyplot as plt


def test_prior_decay():
    a1 = random.uniform(0,1)
    b1 = random.uniform(2,10)

    a2 = random.uniform(0,10)
    b2 = random.uniform(2,50)



    for i in range(100):



