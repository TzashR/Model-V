'''
Functions used on data points. Useful when generating data,training the model and when predicting
'''

import math
import random

import numpy as np
import scipy.stats._continuous_distns

# True calculations - what the simulations uses, not what "we" know


def calculate_point_discrete(target, neighbors,
                             infection_odds=lambda x: random.uniform(0, 1) < math.exp(-0.005 * x),
                             infector_effect=0.03):
    '''Calculates the target's value based on the set of sources using discrete function
        This is a simplification and we use it to start off
    :param target: Datapoint to get the value
    :param neighbors: Datapoint's neighbors
    :param infection_odds:A function that returns true if the neighbor infects the target
    :return: The new value of the datapoint's s.
    '''
    infectors = 0
    if target.s > 0:  # if target is infected, things can get worse even without external infection
        infectors += 1
    for neighbor in neighbors:
        if neighbor.s > 0:  # only infected neighbors can infect
            dist = target.calc_distance(neighbor)
            if infection_odds(dist):
                infectors += 1
    infection_factor = infector_effect * infectors
    new_s = min(target.s + infection_factor, 1)
    return new_s




def fit_average_posterior(dist1_params: tuple, dist2_params: tuple, dist1_type, dist2_type, weights, result_dist_type,
                          n_samples=10000):
    '''
    Takes two distributions, samples from both, for each observation calculates weighted avreage according to weights
    and fits a new distribution
    :param dist1_params: Parameters for first distribution. Should be a tuple
    :param dist2_params: Same as dist1
    :param dist1_type: type of dist1. Should have rvs method
    :param dist2_type: type of dist2. Should have rvs method
    :param weights: a tuple stating how much weight to give each distribution
    :param result_dist_type: A type of distribution e.g. gamma
    :param n_samples:  how many samnples to take from each distribution
    :return: Distribution params, using the result_dist_type fit method.
    '''
    assert (sum(weights) == 1), f"weights should add up to 1. Provided weights are {weights}"

    dist1_samples = dist1_type.rvs(*dist1_params, size=n_samples)  # returns a vector of n_samples samples
    dist2_samples = dist2_type.rvs(*dist2_params, size=n_samples)

    weighted_samples = weights[0] * dist1_samples + weights[1] * dist2_samples
    return result_dist_type.fit(weighted_samples)


def make_kernel(alpha, beta, time_limit=5):
    '''
    Creates the function that will be used as the kernel for points effecting each other
    :param alpha: Decay in distance parameter
    :param beta: Decay in time parameter
    :param time_limit: Max days after which weight is 0
    :return: Function  that takes in observation and a point and calculates observations weight in affecting the point
    '''

    def calc_weight(distance: float, time_diff: int, dist_decay=alpha, time_decay=beta):
        '''
        :param distance: distance between the points
        :param time_diff: time difference current_time - report_time
        :param dist_decay:  parameter for decaying over distance
        :param time_decay:  parameter for decaying over time
        :return: source's weight in predicting target's s
        '''
        assert time_diff >= 0
        if time_diff >= time_limit: return 0
        try:
            weight = math.exp(-time_decay * time_diff - dist_decay * distance)
        except OverflowError:
            print('woosh')
            print(weight)
        return weight
    return calc_weight


def calc_adj_mat(points):
    '''
    Calculates adjacency matrix
    :param points: ndarray of points
    :return: ndarray that is the adjacency matrix
    '''
    n = len(points)
    res_mat = np.zeros(shape=(n, n))
    for i in range(n):
        # the weight formula calculated on the entire row:
        res_mat[i] = np.linalg.norm(points - points[i], axis=1)
    np.fill_diagonal(res_mat, 0)
    return res_mat


def plot_dist(dist_type: scipy.stats._continuous_distns, dist_params: tuple):
    '''
    Plots a graph of the distribution
    :param dist_type: distribution type with rvs function
    :param dist_params: params for the distribution
    :return:
    '''
    x = np.linspace(0, 1, 1000)

    a, loc, scale = dist_params
    dist = dist_type(*dist_params)

    y = dist.pdf(x)
    plt.plot(x, y)
    plt.title(
        f'a = {round(a, 2)}, scale = {round(scale, 2)}, loc = {round(loc, 2)}, mean = {round(a * scale, 2)}, SD = {round(a * (scale ** 2), 2)}')
    plt.show()


def get_range_from_dist(dist_type: scipy.stats._continuous_distns, dist_params: tuple, percentiles=(30, 70)) -> tuple:
    '''
    :param dist_type: a scipy dist type (must have rvs method), e.g gamma
    :param dist_params: tuple with the dist params
    :param percentiles: desired percentiles to be calculated
    :return: a tuple with lower bound and upper bound, the bounds a the percentiles provided
    '''
    return tuple(np.percentile(dist_type.rvs(*dist_params, 1000), percentiles))


