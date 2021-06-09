'''
Generates a data set for the model to learn from
'''

import math
import random as random

from scipy.stats import gamma

from Field_Points import FieldPoint, make_kernel


def main():
    # Define parameters for the data creation
    n_total = 10  # number of points to be created
    observed = 0.8  # percentage of points to be observations
    sources = 0.2  # percentage of points to be sources (with high values of s)
    assert (0 < observed < 1 and 0 <= sources < 1)
    n_obs = math.floor(observed * 1000)
    n_target = n_total-n_obs

    prior = gamma(a=0.5, scale=0.2)  # prior belief distribution for predicted variable
    source_dist = lambda: random.uniform(0.5, 1)  # function that generates values for

    noise = lambda: random.gauss(0, 0.15)  # noise function for created point's predicted value

    # Choose parameters for decay function and create the weight function
    dist_decay_coefficient = 1
    time_decay_coefficient = 1
    calc_weight = make_kernel(dist_decay_coefficient, time_decay_coefficient)

    # ranges of possilbe values for the attributes of points. The following are specific for Fieldpoints
    t_lower = 0
    t_upper = 48

    x_lower = 0
    x_upper = 1000

    y_lower = 0
    y_upper = 1000

    gen_t = lambda: random.uniform(t_lower, t_upper)
    gen_x = lambda: random.uniform(x_lower, x_upper)
    gen_y = lambda: random.uniform(y_lower, y_upper)

    # generate data points
    data_points = []
    target_points = []
    obvservations = []
    for i in range(n_total):
        new_point = FieldPoint(x = gen_x(),y = gen_y(),t = gen_t(),s = prior.rvs())
        data_points.append(new_point)

    # make part of points sources
    for i in range(math.floor(n_total*sources)):
        data_points[i].s = source_dist()

    print(data_points)
if __name__ == "__main__":
    main()
