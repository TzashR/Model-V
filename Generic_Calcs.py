'''
Functions used on data points. Useful when generating data,training the model and when predicting
'''

import math
import random
import numpy as np


#True calculations - wat the simulations uses, not what "we" know
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
            dist = target.calc_dist(neighbor)
            if infection_odds(dist):
                infectors += 1
    infection_factor = infector_effect * infectors
    new_s = min(target.s + infection_factor, 1)
    return new_s


def calculate_point_weights(target, neighbors, weight_func):
    ''' #TODO DO I need this?
    Calculates the target's value based on the set of sources using weights function
    :param target: point where value should be predicted
    :param sources: A list of Datapoints that are the target's neighbors.
    :param weight_func: A weight function for the type of the points. The
    function should recieve (source, target)
    :return: Predicted value (depends on the type of data points)
    '''
    weights_sum = 0
    weighted_values_sum = 0
    for observation, val in neighbors:
        weight = weight_func(observation, target)
        weights_sum += weight
        weighted_values_sum += weight * val
    return weighted_values_sum / weights_sum




def predict_point(target, neighbors, weight_func, priors, dist_type, dist_func, k=100):
    '''
    :param target: Target Datapoint
    :param neighbors:  Target's neighbors
    :param weight_func: the weight func.
    :param priors: dictionary of priors for each point
    :param dist_type: dist object from scipy. e.g gamma
    :param dist_func: a function that takes the relevant parameters and returns rvs.
    e.g lambda a,b: gamma.rvs(alpha = a, beta = b)
    :param k:
    :return:
    '''
    # TODO use the veracity
    weights = np.array([weight_func(neighbor, target) for neighbor in neighbors])
    weights_sum = sum(weights)
    predictions = []

    for i in range(k):
        values = []
        for neighbor in neighbors:
            alpha, beta = priors[neighbor]
            values.append(dist_func(alpha, beta))
        predicted = weights.dot(np.array(values)) / weights_sum
        predictions.append(predicted)
    new_dist = dist_type.fit(predictions)
    return new_dist




def make_kernel(alpha, beta, time_limit=5):
    '''
    Creates the function that will be used as the kernel for points effecting each other
    :param alpha: Decay in distance parameter
    :param beta: Decay in time parameter
    :param time_limit: Max days after which weight is 0
    :return: Function  that takes in observation and a point and calculates observations weight in affecting the point
    '''

    def calc_weight(source, target, dist_decay=alpha, time_decay=beta):
        '''
        :param source: source observation (Observation)
        :param target: target point (DataPoint)
        :return: source's weight in predicting target's s
        '''
        assert source.t < target.t
        time_diff = target.t - source.t
        if time_diff >= time_limit: return 0
        dist = source.calc_dist(target)
        return math.exp(-time_decay * time_diff - dist_decay * dist)

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
